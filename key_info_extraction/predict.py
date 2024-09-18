import torch
from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from pathlib import Path
from tqdm import tqdm
from config import kie_model_dir, kie_result_dir, kie_boxes_transcripts_temp, rec_thresh
import key_info_extraction.PICK.model.pick as pick_arch_module
from key_info_extraction.PICK.data_utils.pick_dataset import PICKDataset
from key_info_extraction.PICK.data_utils.pick_dataset import BatchCollateFn
from key_info_extraction.PICK.utils.util import iob_index_to_str, text_index_to_str

from allennlp.data.dataset_readers.dataset_utils.span_utils import InvalidTagSequence
from typing import List, Tuple
TypedStringSpan = Tuple[str, Tuple[int, int]]


def bio_tags_to_spans2(
        tag_sequence: List[str], text_length: List[int] = None
) -> List[TypedStringSpan]:
    list_idx_to_split = [0]
    init_idx = 0
    for text_len in text_length[0]:
        init_idx += text_len
        list_idx_to_split.append(init_idx)

    spans = []
    line_pos_from_bottom = []
    for index, string_tag in enumerate(tag_sequence):
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]

        if bio_tag == "B":
            if index in list_idx_to_split:
                idx_start = list_idx_to_split.index(index)
                idx_end = list_idx_to_split[idx_start + 1] - 1
                spans.append((conll_tag, (index, idx_end)))
                line_pos_from_bottom.append(idx_start)
    return spans, line_pos_from_bottom


class KeyInfoExtractor:
    def __init__(self, checkpoint_path=kie_model_dir, gpu_id=-1):
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        # Thiết lập thiết bị (CPU hoặc GPU)
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
        # Load mô hình từ checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint['config']
        self.state_dict = self.checkpoint['state_dict']

        # Khởi tạo mô hình
        self.pick_model = self.config.init_obj('model_arch', pick_arch_module)
        self.pick_model = self.pick_model.to(self.device)
        self.pick_model.load_state_dict(self.state_dict)
        self.pick_model.eval()

    def extract(self, images_folder, boxes_and_transcripts, batch_size=1):
        # setup dataset and data_loader instances
        test_dataset = PICKDataset(boxes_and_transcripts_folder=boxes_and_transcripts,
                                   images_folder=images_folder,
                                   resized_image_size=(480, 960),
                                   ignore_error=False,
                                   training=False)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=0, collate_fn=BatchCollateFn(training=False))

        # setup output path
        output_path = Path(kie_result_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # res = []
        # predict and save to file
        with torch.no_grad():
            for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(self.device)

                # For easier debug.
                image_names = input_data_item["filenames"]

                output = self.pick_model(**input_data_item)
                logits = output['logits']  # (B, N*T, out_dim)
                new_mask = output['new_mask']
                image_indexs = input_data_item['image_indexs']  # (B,)
                text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
                text_length = input_data_item['text_length']
                mask = input_data_item['mask']
                # List[(List[int], torch.Tensor)]
                best_paths = self.pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                # convert iob index to iob string
                decoded_tags_list = iob_index_to_str(predicted_tags)
                # union text as a sequence and convert index to string
                decoded_texts_list = text_index_to_str(text_segments, mask)

                for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list,
                                                                    image_indexs):
                    # List[ Tuple[str, Tuple[int, int]] ]
                    # spans = bio_tags_to_spans(decoded_tags, [])
                    # spans = sorted(spans, key=lambda x: x[1][0])
                    spans, line_pos_from_bottom = bio_tags_to_spans2(decoded_tags, text_length.cpu().numpy())

                    entities = {}  # exists one to many case
                    for entity_name, range_tuple in spans:
                        text = ''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1])
                        if entity_name not in entities:
                            entities[entity_name] = text
                        else:
                            entities[entity_name] += ' - ' + text
                    return entities
                    # result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')

                    # res.append((result_file, entities))

                    # with result_file.open(mode='w', encoding='utf8') as f:
                    #     for item in entities:
                    #         f.write('{}\t{}\n'.format(item['entity_name'], item['text']))

    def save_boxes_and_transcripts(self, img_path, boxes_list, txts, scores):
        img_name = Path(img_path).stem
        output_path = Path(kie_boxes_transcripts_temp)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path.joinpath(f'{img_name}.tsv')

        res = ''
        for idx, box in enumerate(boxes_list):
            str_box = ','.join(map(str, box))
            if scores[idx] > rec_thresh:
                res += f"{idx+1},{str_box},{txts[idx]}"
            else:
                res += f"{idx+1},{str_box},"
            res += '\n'
        res = res.rstrip('\n')
        with output_file.open(mode='w', encoding='utf8') as f:
            f.write(res)


