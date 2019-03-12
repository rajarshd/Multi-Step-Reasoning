from model.retriever_module import LSTMParagraphScorer
import logging


logger = logging.getLogger()


class LSTMRetriever():
    def __init__(self, args, word_dict, feature_dict):

        self.args = args
        self.word_dict = word_dict
        self.feature_dict = feature_dict
        self.model = LSTMParagraphScorer(args, word_dict, feature_dict)
        if self.args.cuda:
            self.model = self.model.cuda()

    def get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def score_paras(self, paras, para_mask, query, query_mask):

        scores, doc, ques = self.model(paras, para_mask, query, query_mask)
        return scores, doc.cpu().data.numpy(), ques.cpu().data.numpy()