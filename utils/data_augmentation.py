from transformers import BertTokenizer, PreTrainedTokenizer, PreTrainedModel
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from typing import List, Tuple, Set

class BertBaseAugmentation:
    def __init__(self, tokenizer_name: str = 'bert-base-multilingual-cased', model_name: str = 'jhgan/ko-sroberta-multitask'):
        """
        BertBaseAugmentation 클래스 초기화 메서드.
        주어진 토크나이저와 모델 이름을 사용하여 tokenizer와 model을 로드합니다.

        Args:
            tokenizer_name (str): 사용할 토크나이저의 이름. 기본값은 'bert-base-multilingual-cased'.
            model_name (str): 사용할 모델의 이름. 기본값은 'jhgan/ko-sroberta-multitask'.
        """
        self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model: PreTrainedModel = SentenceTransformer(model_name)

    def _generate_masked_text(self, input_ids: List[int]) -> str:
        """
        입력된 토큰 ID 리스트를 기반으로 마스킹된 텍스트를 생성합니다.

        Args:
            input_ids (List[int]): 토큰 ID의 리스트.

        Returns:
            str: 마스킹된 텍스트.
        """
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

    def augment(self, text: str, aug_num: int = 10, similarity_threshold: float = 0.8) -> Tuple[List[str], List[float]]:
        """
        주어진 텍스트를 기반으로 aug_num 개수만큼의 증강된 텍스트를 생성하고,
        원본 텍스트와 각 증강된 텍스트 간의 코사인 유사도를 계산합니다.
        유사도가 similarity_threshold 이상이며 중복되지 않는 텍스트만을 반환합니다.

        Args:
            text (str): 원본 텍스트.
            aug_num (int): 생성할 증강 텍스트의 개수. 기본값은 10.
            similarity_threshold (float): 유사도 임계값. 기본값은 0.8.

        Returns:
            Tuple[List[str], List[float]]: 증강된 텍스트 리스트와 각 텍스트의 유사도 리스트.
        """
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        original_embedding = self.model.encode(text, convert_to_tensor=True)

        augmented_texts: List[str] = []
        similarities: List[float] = []
        generated_texts: Set[str] = set()

        while len(augmented_texts) < aug_num:
            new_text = self._generate_masked_text(input_ids)

            if new_text in generated_texts:
                continue

            new_embedding = self.model.encode(new_text, convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(original_embedding, new_embedding).item()

            if cosine_score >= similarity_threshold:
                augmented_texts.append(new_text)
                similarities.append(cosine_score)
                generated_texts.add(new_text)

        return augmented_texts, similarities

# 사용 예시
if __name__ == "__main__":
    augmenter = BertBaseAugmentation()
    original_text = "한국어 텍스트를 BERT 모델로 마스킹합니다."
    augmented_texts, similarities = augmenter.augment(original_text, aug_num=10, similarity_threshold=0.85)

    for i, (aug_text, sim) in enumerate(zip(augmented_texts, similarities), 1):
        print(f"{i}: {aug_text} (유사도: {sim:.4f})")
