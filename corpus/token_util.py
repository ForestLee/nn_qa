from .tokenizer import Tokenizer
import numpy as np

class TokenUtil:
    def __init__(self, token_dict, ner_vocab=False):
        self.tokenizer = Tokenizer(token_dict, ner_vocab=ner_vocab)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_inputs_indexs(self, texts, max_len):
        """获取模型的预测输入
        """
        tokens, segs, token_lens = [], [], []
        for text in texts:
            token, seg = self.tokenizer.encode(text, first_length=max_len)
            tokens.append(np.array(token))
            #segs.append(np.array(seg))
            token_lens.append(len(text))

        return tokens, token_lens

    def get_input_indexs(self, text, max_len):
        """获取模型的预测输入
        """
        token, seg = self.tokenizer.encode(text, first_length=max_len)
        return np.array(token)