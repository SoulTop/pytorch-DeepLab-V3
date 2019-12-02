from dataloaders.datasets import pascal
from torch.utils.data import DataLoader

def make_data_loader(basedir, args, **kwargs):
    if args.dataset == 'pascal':
        if args.pattern == 'train':
            train_set = pascal.VOCSegmentation(args, base_dir=basedir, split='train')
            val_set = pascal.VOCSegmentation(args, base_dir=basedir, split='val')

            num_class = train_set.NUM_CLASSES

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            return train_loader, val_loader, num_class
        if args.pattern == 'test':
            test_set = pascal.VOCSegmentation(args, base_dir=basedir, split='test')

            num_class = test_set.NUM_CLASSES

            test_set_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
            return test_set_loader, num_class
    else:
        raise NotImplementedError

