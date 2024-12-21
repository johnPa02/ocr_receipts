import torch

class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()

        pred_idxs = preds.argmax(axis=1)
        # print('1', preds)
        # print('pred_idxs',pred_idxs)
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = label.argmax(1)
        label = [(self.label_list[idx], 1.0) for idx in label]
        # print(decode_out, label)
        return decode_out, label