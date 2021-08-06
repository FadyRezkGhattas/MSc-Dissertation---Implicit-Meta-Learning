from DatasetsUtil.DatasetsCreator import DATASET_GETTERS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class DataLoaderWrap():
    def __init__(self, args, train_sampler, labeled_epoch, unlabeled_epoch) -> None:
        super().__init__()

        self.args = args

        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.data](args)
        self.labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=train_sampler(labeled_dataset),
            batch_size=args.labelled_batch_size,
            drop_last=True)

        self.unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=args.labelled_batch_size*args.mu,
            drop_last=True)

        self.labeled_epoch = labeled_epoch
        self.unlabeled_epoch = unlabeled_epoch

        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=test_dataset.__len__())

        self.labeled_iter = iter(self.labeled_trainloader)
        self.unlabeled_iter = iter(self.unlabeled_trainloader)
        
        test_data_full = next(iter(test_loader))

        self.val_data_x = test_data_full[0][0:args.validation_size]
        self.val_data_y = test_data_full[1][0:args.validation_size]

        self.test_data_x = test_data_full[0][args.validation_size:]
        self.test_data_y = test_data_full[1][args.validation_size:]
    
    def get_labelled_batch(self):
        try:
            inputs_x, targets_x = self.labeled_iter.next()
        except:
            self.labeled_iter = iter(self.labeled_trainloader)
            inputs_x, targets_x = self.labeled_iter.next()
        return inputs_x, targets_x
    
    def get_unlabeled_batch(self):
        try:
            (inputs_u, inputs_u_w, inputs_u_s), _ = self.unlabeled_iter.next()
        except:
                self.unlabeled_iter = iter(self.unlabeled_trainloader)
                (inputs_u, inputs_u_w, inputs_u_s), _ = self.unlabeled_iter.next()
        return inputs_u, inputs_u_w, inputs_u_s