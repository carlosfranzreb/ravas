import torch
import glob
from stream_processing.Processor import ProcessingCallback


class KNNVC(ProcessingCallback):
    def __init__(self, ref_dir):
        super().__init__()
        self.ref_dir = ref_dir

    def init_callback(self):
        knn_vc = torch.hub.load(
            "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True
        )
        path = self.ref_dir
        ref_wav_paths = glob.glob(path + "/*.flac")

        matching_set = knn_vc.get_matching_set(ref_wav_paths)
        history = []
        time_history = []

        return [knn_vc, matching_set, history, time_history]

    def callback(self, dtime, data, knn_vc, matching_set, history, time_history):
        # int16 to float32
        y = data.to(torch.float32)
        y = y / 32768.0

        # append to history
        history.append(y)
        time_history.append(dtime)

        # only process if we have enough history
        if len(history) < 3:
            return None
        input = torch.cat(history, dim=0)
        tr = torch.cat(time_history, dim=0)
        sizes = [len(x) for x in history]

        if torch.max(torch.abs(input)) < 0.1:
            out = torch.zeros_like(data)

        else:
            query_seq = knn_vc.get_features(input)
            out = knn_vc.match(query_seq, matching_set, topk=4)
            # get middle part of output
            out = out[sizes[0] : sizes[0] + sizes[1]]
        # get middle part of time
        out_time = tr[sizes[0] : sizes[0] + sizes[1]]

        history.pop(0)
        time_history.pop(0)

        # float32 to int16
        out = torch.clamp(out, -1.0, 1.0)
        out = out * 32768.0
        out = out.to(torch.int16)
        return out_time, out
