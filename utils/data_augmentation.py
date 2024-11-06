import logging
from transformers import BertTokenizer, PreTrainedTokenizer, PreTrainedModel
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
from typing import List, Tuple, Set
import time

# 로깅 설정: augmentation.log 파일에 로그 기록
logging.basicConfig(
    filename='augmentation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

class BertBaseAugmentation:
    def __init__(self, tokenizer_name: str = 'bert-base-multilingual-cased', model_name: str = 'jhgan/ko-sroberta-multitask'):
        start_time = time.time()
        self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model: PreTrainedModel = SentenceTransformer(model_name)
        logger.info(f"토크나이저와 모델 로드 완료 - 경과 시간: {time.time() - start_time:.2f}초")

    def _generate_masked_text(self, input_ids: List[int]) -> str:
        special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id}
        candidate_indices = [i for i, token_id in enumerate(input_ids) if token_id not in special_tokens]
        num_to_mask = max(1, int(len(candidate_indices) * 0.15))
        masked_indices = random.sample(candidate_indices, num_to_mask)
        masked_input_ids = input_ids[:]

        for idx in masked_indices:
            prob = random.random()
            if prob < 0.8:
                masked_input_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                masked_input_ids[idx] = random.randint(0, self.tokenizer.vocab_size - 1)

        return self.tokenizer.decode(masked_input_ids, skip_special_tokens=True)

    def augment(self, text: str, aug_num: int = 10, similarity_threshold: float = 0.7, max_attempts: int = 5000) -> Tuple[List[str], List[float]]:
        """
        주어진 텍스트를 기반으로 aug_num 개수만큼의 증강된 텍스트를 생성합니다.
        무한 루프 방지를 위해 max_attempts 매개변수를 증가시킵니다.
        
        Args:
            text (str): 원본 텍스트.
            aug_num (int): 생성할 증강 텍스트의 개수. 기본값은 10.
            similarity_threshold (float): 유사도 임계값. 기본값은 0.8.
            max_attempts (int): 최대 시도 횟수. 기본값은 5000.
        
        Returns:
            Tuple[List[str], List[float]]: 증강된 텍스트 리스트와 각 텍스트의 유사도 리스트.
        """
        logger.info("텍스트 증강 시작...")
        start_time = time.time()

        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        original_embedding = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        augmented_texts = []
        similarities = []
        generated_texts = set()

        attempts = 0

        while len(augmented_texts) < aug_num:
            if attempts >= max_attempts:
                logger.warning(f"최대 시도 횟수 {max_attempts}에 도달하여 증강을 중단합니다.")
                break

            new_text = self._generate_masked_text(input_ids)
            attempts += 1

            if new_text in generated_texts:
                continue

            new_embedding = self.model.encode(new_text, convert_to_tensor=True, show_progress_bar=False)
            cosine_score = util.pytorch_cos_sim(original_embedding, new_embedding).item()

            if cosine_score >= similarity_threshold:
                augmented_texts.append(new_text)
                similarities.append(cosine_score)
                generated_texts.add(new_text)
                logger.info(f"유효한 증강 텍스트 생성: {new_text} (유사도: {cosine_score:.4f})")

        logger.info(f"텍스트 증강 완료 - 총 시도 횟수: {attempts}, 총 경과 시간: {time.time() - start_time:.2f}초")
        return augmented_texts, similarities
    