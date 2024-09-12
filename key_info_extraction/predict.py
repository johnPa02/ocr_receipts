import torch
from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

import PICK.model.pick as pick_arch_module
from PICK.data_utils.pick_dataset import PICKDataset
from PICK.data_utils.pick_dataset import BatchCollateFn
from PICK.utils.util import iob_index_to_str, text_index_to_str


class KeyInfoExtractor:
    def __init__(self, checkpoint_path, gpu_id=-1):
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

    def extract(self, images_folder, boxes_and_transcripts):
        # Tạo một danh sách dữ liệu đầu vào cho ảnh
        test_dataset = PICKDataset(
            boxes_and_transcripts_folder=boxes_and_transcripts,
            images_folder=images_folder,
            resized_image_size=(480, 960),
            ignore_error=False,
            training=False
        )

        # Chỉ tạo DataLoader cho một ảnh
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=BatchCollateFn(training=False)
        )

        # Dự đoán và trích xuất thông tin
        with torch.no_grad():
            for input_data_item in test_data_loader:
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(self.device)

                output = self.pick_model(**input_data_item)
                logits = output['logits']
                new_mask = output['new_mask']
                text_segments = input_data_item['text_segments']
                mask = input_data_item['mask']

                # Lấy các nhãn dự đoán
                best_paths = self.pick_model.decoder.crf_layer.viterbi_tags(
                    logits, mask=new_mask, logits_batch_first=True)
                predicted_tags = [path for path, score in best_paths]

                # Chuyển đổi nhãn IOB sang chuỗi văn bản
                decoded_tags_list = iob_index_to_str(predicted_tags)
                decoded_texts_list = text_index_to_str(text_segments, mask)

                entities = []
                for decoded_tags, decoded_texts in zip(decoded_tags_list, decoded_texts_list):
                    spans = bio_tags_to_spans(decoded_tags, [])
                    spans = sorted(spans, key=lambda x: x[1][0])

                    # Lưu các thực thể (entities) trích xuất được
                    for entity_name, range_tuple in spans:
                        entity = {
                            'entity_name': entity_name,
                            'text': ''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1])
                        }
                        entities.append(entity)
                return entities
