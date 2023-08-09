import torch
from torch.cuda import amp
from torchmetrics import MeanMetric, Accuracy

from contextlib import nullcontext
from tqdm import tqdm

from utils import segmentation_metrics, adaptive_clip_grad


class SegmentationTrainer:
    def __init__(self, model, optimizer, criterion,
                 lr_scheduler=None, use_cuda=None, use_amp=True, use_ema=False):
        ### set models ###
        self.model = model
        self.use_ema = use_ema
        self.avg_model = ModelEMA(self.model) if self.use_ema else model
        ##################

        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self._train_count = 0

        self._initialize_cuda(use_cuda, use_amp)
        self.grad_scaler = amp.GradScaler(enabled=self.use_amp)

        ### define metrics ###
        self.metrics = {
            'loss': MeanMetric(),
            'pixel_acc': MeanMetric(),
            'iou': MeanMetric(),
            'dice': MeanMetric(),
            'precision': MeanMetric(),
            'recall': MeanMetric(),
        }
        ######################

    def fit(self, train_loader, val_loader=None, epochs=1, verbose=True):
        history = {name: [] for name in self.metrics.keys()}
        if val_loader is not None:  # return history for validation data
            history = {**history,
                       **{'val_' + name: [] for name in self.metrics.keys()}}

        for epoch in range(1, epochs + 1):
            ### model setting ###
            if self.use_cuda:
                self.model.cuda()
                self.avg_model.cuda()
                self.criterion.cuda()
            self.model.train()
            ######################

            # train_loader.sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            if verbose:
                train_iter = tqdm(train_iter, total=len(train_loader))
                train_iter.set_description(f'Epoch {epoch}/{epochs}')
            self._reset_metrics()

            for step, data in enumerate(train_iter):
                self.train_step(data)
                if verbose:  # show metrics on train data
                    train_iter.set_postfix(self._compute_metrics())
            self._update_history(history)

            if val_loader is not None:
                self.evaluate(val_loader, verbose=verbose)
                self._update_history(history, prefix='val_')
        return history

    def evaluate(self, val_loader, verbose=True):
        ### model setting ###
        if self.use_cuda:
            self.model.cuda()
            self.avg_model.cuda()
            self.criterion.cuda()
        self.model.eval()
        self.avg_model.eval()
        #####################

        val_iter = iter(val_loader)
        if verbose:
            val_iter = tqdm(val_iter, total=len(val_loader))
            val_iter.set_description(f'Eval')

        self._reset_metrics()

        for step, data in enumerate(val_iter):
            self.test_step(data)
            if verbose:
                val_iter.set_postfix(self._compute_metrics())  # show metrics on val data
        return self._compute_metrics()

    def predict(self, data_loader, verbose=True):
        ### model setting ###
        if self.use_cuda:
            self.model.cuda()
            self.avg_model.cuda()
        self.model.eval()
        self.avg_model.eval()
        #####################

        data_iter = iter(data_loader)
        if verbose:
            data_iter = tqdm(data_iter, total=len(data_loader))

        results = []
        for step, data in enumerate(data_iter):
            outputs = self.predict_step(data)
            results.append(outputs)

        if isinstance(outputs, (tuple, list)):  # multi outputs
            results = list(*zip(results))
            results = [torch.cat(tensor, dim=0) for tensor in results]
        else:
            results = torch.cat(results, dim=0)
        return results

    def train_step(self, data):
        ### config input data ###
        data = self._convert_cuda_data(data)
        x, y = data
        #########################

        ### optimizer.zero_grad() ###
        self.optimizer.zero_grad()
        #############################
        with self.autocast():
            ### forward pass ###
            logits = self.model(x)
            loss = self.criterion(logits, y)
            ####################

        ### update model ###
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        adaptive_clip_grad(self.model.parameters())
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.use_ema:
            self.avg_model.update_parameters(self.model)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self._train_count += 1
        #####################

        ### update metrics ###
        y, logits, loss = self._convert_not_training_data([y, logits, loss])
        pixel_acc, iou, dice, precision, recall = segmentation_metrics(logits.float(), y)

        self.metrics['pixel_acc'].update(pixel_acc)
        self.metrics['iou'].update(iou)
        self.metrics['dice'].update(dice)
        self.metrics['precision'].update(precision)
        self.metrics['recall'].update(recall)
        self.metrics['loss'].update(loss)

        return

    def test_step(self, data):
        ### config input data ###
        data = self._convert_cuda_data(data)
        x, y = data
        #########################

        with self.autocast():
            with torch.no_grad():
                ### forward pass ###
                logits = self.avg_model(x)
                loss = self.criterion(logits, y)
                ####################

        ### update metrics ###
        y, logits, loss = self._convert_not_training_data([y, logits, loss])
        pixel_acc, iou, dice, precision, recall = segmentation_metrics(logits.float(), y)

        self.metrics['pixel_acc'].update(pixel_acc)
        self.metrics['iou'].update(iou)
        self.metrics['dice'].update(dice)
        self.metrics['precision'].update(precision)
        self.metrics['recall'].update(recall)
        self.metrics['loss'].update(loss)

        #######################
        return

    def predict_step(self, data):
        ### config input data ###
        data = self._convert_cuda_data(data)
        x = data
        #########################

        with self.autocast():
            with torch.no_grad():
                ### forward pass ###
                logits = self.avg_model(x)
                ####################

        return logits.detach().cpu()

    def _initialize_cuda(self, use_cuda=None, use_amp=True):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.use_amp = use_amp and self.use_cuda
        self.autocast = amp.autocast if self.use_amp else nullcontext

    def _convert_not_training_data(self, data, detach=True, cpu=True, numpy=False):
        if isinstance(data, (tuple, list)):
            return [self._convert_not_training_data(e) for e in data]
        elif isinstance(data, dict):
            return {k: self._convert_not_training_data(v) for k, v in data.items()}
        else:
            if detach:
                data = data.detach()
            if cpu:
                data = data.cpu()
            if numpy:
                data = data.numpy()
            return data

    def _convert_cuda_data(self, data):
        if not self.use_cuda:
            return data
        if isinstance(data, (tuple, list)):
            return [self._convert_cuda_data(e) for e in data]
        elif isinstance(data, dict):
            return {k: self._convert_cuda_data(v) for k, v in data.items()}
        else:
            return data.cuda()

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def _update_history(self, history, prefix=''):
        metrics = self._compute_metrics()
        for name, value in metrics.items():
            history[prefix + name].append(value)