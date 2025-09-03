import torch.nn as nn
import torch
from utils.utils_ import *

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_cls_ext(args, net):

    if args.arch == 'tanet':
        classifier = net.module.new_fc
        ext = net
        ext.module.new_fc = nn.Identity()
        for k, v in classifier.named_parameters():
            v.requires_grad = False
    elif args.arch == 'videoswintransformer':
        # Freeze the classifier head parameters
        for k, v in net.module.cls_head.named_parameters():
            v.requires_grad = False
        # Use the final linear layer of the classifier head as the classifier
        classifier = net.module.cls_head.fc_cls
        # Build feature extractor: backbone -> adaptive pooling -> flatten
        ext = nn.Sequential(
            net.module.backbone,
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1)
        )
    else:
        for k, v in net.named_parameters():
            if 'logits' in k:
                v.requires_grad = False  # freeze the  classifier
        classifier = nn.Sequential(*list(net.module.logits.children()))
        ext = list(net.module.children())[3:] + list(net.module.children())[:2]
        ext = nn.Sequential(*ext)

    return ext, classifier


class T3A(nn.Module):
    """
    Test Time Template Adjustments (T3A)
    """

    def __init__(self, args, ext, classifier):
        super().__init__()
        self.args = args
        self.model = ext
        self.classifier = classifier
        
        # Handle different classifier types
        if hasattr(classifier, 'weight'):
            # Standard linear classifier (now including Video Swin Transformer's linear layer)
            self.warmup_supports = self.classifier.weight.data
        else:
            # Fallback: use random initialization
            feature_dim = ext.output_dim if hasattr(ext, 'output_dim') else 512
            self.warmup_supports = torch.randn(args.num_classes, feature_dim)
        
        # Compute class probabilities without passing weight tensors
        warmup_prob = torch.softmax(
            torch.mm(self.warmup_supports, self.warmup_supports.t()), 
            dim=1
        )
        
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=args.num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.t3a_filter_k
        self.num_classes = args.num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        with torch.no_grad():
            z = self.model(x)
        # online adaptation
        p = self.classifier(z)
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        # prediction
        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        device = self.supports.device  # Get device from model parameters
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long().to(device)  # Ensure y_hat is on the same device
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.arange(len(ent_s), device=device)
            return self.supports[indices], self.labels[indices]

        indices = []
        indices1 = torch.arange(len(ent_s), device=device)  # Create tensor directly on the right device
        for i in range(self.num_classes):
            mask = (y_hat == i)
            if mask.any():
                _, indices2 = torch.sort(ent_s[mask])
                if filter_K > 0 and len(indices2) > 0:
                    selected = indices1[mask][indices2][:filter_K]
                    indices.append(selected)
        
        if not indices:  # Handle case where no indices were selected
            return self.supports, self.labels
            
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels


def t3a_forward_and_adapt(args, ext, cls, val_loader):
    model = T3A(args, ext, cls)
    with torch.no_grad():
        total = 0
        correct_list = []
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        for i, (input, target) in enumerate(val_loader):  #
            ext.eval()
            cls.eval()
            actual_bz = input.shape[0]
            n_views = input.shape[1]
            input = input.cuda()
            target = target.cuda()
            if args.arch == 'tanet':
                n_clips = int(args.sample_style.split("-")[-1])
                input = input.view(-1, 3, input.size(2), input.size(3))
                input = input.view(actual_bz * args.test_crops * n_clips,
                                       args.clip_length, 3, input.size(2), input.size(3))
                output = model(input)
                output = output.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)
                # Ensure we maintain batch dimension for accuracy calculation
                if output.dim() == 1:
                    logits = output.unsqueeze(0)
                else:
                    logits = output
                prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
                top1.update(prec1.item(), actual_bz)
                top5.update(prec5.item(), actual_bz)
            elif args.arch == 'videoswintransformer':
                input = input.reshape((-1,) + input.shape[2:])
                output = model(input)
                # Ensure we maintain batch dimension for accuracy calculation
                if output.dim() == 1:
                    logits = output.unsqueeze(0)
                else:
                    logits = output
                prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
                top1.update(prec1.item(), actual_bz)
                top5.update(prec5.item(), actual_bz)
            else:
                input = input.reshape((-1,) + input.shape[2:])
                output = model(input)
                logits = torch.squeeze(output)
                prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
                top1.update(prec1.item(), actual_bz)
                top5.update(prec5.item(), actual_bz)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, top1=top1, top5=top5))
    
    return top1.avg
