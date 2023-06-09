import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
import numpy as np

from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import dataset.incremental_dataloader

from .utils import build_cosine_scheduler
import pdb
import time

class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, n_ctx=16, prompt_pos=2):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        # pdb.set_trace()

        n_cls = len(class_names)
        self.dtype = dtype
        ctx_vectors = torch.empty(1, n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        # pdb.set_trace()
        prompt_prefix =' '.join(['x'] * n_ctx)
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]

        classnames = [name.replace('_', ' ') for name in class_names]
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]

        self.prompt_pos = prompt_pos

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype) # 引入SOS EOS
        self.register_buffer( 'token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer( 'token_suffix', embedding[:, 1+n_ctx:,:]) # CLS, EOS, [n_cls, -1, ctx_dim]

        self.n_cls = n_cls 
        self.n_ctx = n_ctx 
        self.ctx_dim = ctx_dim

    def forward(self):
        # pdb.set_trace()
        ctx=self.ctx

        tokenized_prompts = self.tokenized_prompts.view(self.n_cls,-1) #编码后的prompt 100，77

        n_cls = self.n_cls

        if self.prompt_pos == 2:
            prefix = self.token_prefix.unsqueeze(1)
            suffix = self.token_suffix.unsqueeze(1)
            ctx = ctx.unsqueeze(0).repeat(n_cls, 1, 1, 1)
            prompts = torch.cat([prefix, ctx, suffix],dim=2) # SOS+可训练prompt+EOS  embedding
        elif self.prompt_pos == 1:
            prompts =[]
            half_n_ctx = self.n_ctx // 2
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1,:name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i_half1 = ctx[:,:half_n_ctx, :].unsqueeze(0)
                ctx_i_half2 = ctx[:, half_n_ctx:,:].unsqueeze(0)
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.prompt_pos == 0:
            prompts =[]
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1,:,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1, :name_len,:].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        prompts = prompts.view(n_cls, -1, self.ctx_dim)
        self.prompts = prompts
        self.prompts_token = tokenized_prompts
        return prompts, tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype) #position_embeding可训练
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # @ and
        return x


class CLIP(nn.Module):
    def __init__(self, class_names, clip_model, n_ctx=16):
        super().__init__()
        self.n_class = len(class_names)

        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)

        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.prompt_learner = PromptLearner(class_names, clip_model, n_ctx=n_ctx)

        # image encoder
        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale

    def forward(self, image, num_test=None, test_class=None, prototype=None, test=False):

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        if test:
            n_test = len(test_class)
            # prototype = torch.from_numpy(prototype.type(self.dtype)).cuda()
            # prototype = torch.from_numpy(prototype[:,0]).cuda()
            ima_proto = torch.tensor([item for item in prototype[:,0]]).type(self.dtype).cuda()
            probability = image_features @ ima_proto.t()
            _, indices = probability.topk(k=min(num_test,probability.shape[1]), dim=1, largest=True)
            text_prompt = torch.tensor([item for item in prototype[:,1]]).reshape(n_test, 77, 512).cuda()
            tokenized_prompts = torch.tensor([item for item in prototype[:,2]]).reshape(n_test, 77).cuda()
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # text_features = self.text_features
            logit_scale = self.logit_scale.exp()
            logit = image_features @ text_features.t()
            logits = torch.zeros_like(logit)
            for i in range(min(num_test,probability.shape[1])):
                a = indices[:,i]*10
                for j in range(indices.shape[0]):
                    logits[j][a[j]:a[j]+10] = logit[j][a[j]:a[j]+10]
            # pdb.set_trace()
            logits = logit_scale * logits
            return logits

        else:
            n_class = self.n_class
            text_prompt, tokenized_prompts = self.prompt_learner()
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # pdb.set_trace()
            text_features = text_features.view(n_class, -1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()   #128*100

            return logits, image_features

    @torch.no_grad()
    def set_classifier(self):
        text_prompt, tokenized_prompts = self.prompt_learner()
        try:
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
        except:
            text_features = []
            batch_size= 1000
            for bi in range(text_prompt.shape[0]//batch_size):
                batch_text_features = self.text_encoder(text_prompt[bi*1000:(bi+1)*1000], tokenized_prompts[bi*1000:(bi+1)*1000])
                text_features.append(batch_text_features)
            text_features = torch.cat(text_features, dim=0)
        n_dim = text_features.shape[-1]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_class, -1)

        # text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

    @property #变成属性
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype #return int/float


class CoOp:
    def __init__(self, args, n_ctx=16, use_float32=False, use_grad_checkpoint=False):
        clip_model, _ = load(args.ckpt_path)
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint

        self.n_ctx = n_ctx # n_ctx 输入词数
        self.lr = args.lr*args.train_batch/20
        self.wd = args.wd # wd ??
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.args = args
        self.proto = []

    def fit(self, data, len_train):

        train_loader = data['train_loader']
        image_proto = 0
        if len(train_loader.dataset)< self.train_batch:
            real_img_bsz = len(train_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch

        per_epoch_steps = len(train_loader)

        self.init_model(class_names=data['class_names'], per_epoch_steps=per_epoch_steps)

        self.model.eval()

        for epoch in range(self.epochs):
            for idx, (x, y) in enumerate(train_loader):
                # pdb.set_trace()
                y = y - 10 * self.args.sess
                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)

                output,ima_feat = self.model(x.cuda())
                # pdb.set_trace()
                if epoch == self.epochs-1:
                    image_proto += ima_feat.sum(0)

                loss = F.cross_entropy(output, y.cuda())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # pdb.set_trace()
        proto= [image_proto.cpu().detach().numpy() / len_train, self.model.prompt_learner.prompts.cpu().detach().numpy(), self.model.prompt_learner.prompts_token.cpu().detach().numpy()]
        self.proto.append(proto)
        self.prototype = np.array(self.proto, dtype=object)
        # self.prototype = np.array(self.proto)
        # self.model.set_classifier()
        return self.prototype


    def init_model(self, class_names, per_epoch_steps):

        self.n_class = len(class_names)
        clip_model = deepcopy(self.clip_model)

        self.model = CLIP(class_names, clip_model, self.n_ctx)
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True 
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True

        param_dict = [{'params': [p for p in self.model.prompt_learner.parameters() if p.requires_grad]}]
        self.optimizer = torch.optim.SGD(param_dict, lr=self.lr, weight_decay=self.wd)
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=self.lr,
            total_step=self.epochs*per_epoch_steps)

    @torch.no_grad()
    def accuracy(self, loader, num_test, test_class, prototype, mean_per_class=False):
        if mean_per_class:
            return self._accuracy_mpc(loader, num_test, test_class, prototype)
        else:
            return self._accuracy(loader, num_test, test_class, prototype)

    def _accuracy_mpc(self, loader, num_test, test_class, prototype):
        n_class = self.n_class
        acc_per_class = [0 for _ in range(n_class)]
        count_per_class = [0 for _ in range(n_class)]
        for i, (x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            for c in range(n_class):
                acc_per_class[c] += ((top_labels.view(-1) == y.cuda()) * (y.cuda()== c)).sum().item()
                count_per_class[c] += (y.cuda() == c).sum().item()
        acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()
        return acc

    def _accuracy(self, loader, num_test, test_class, prototype):
        total_count=0
        acc_count =0
        # pdb.set_trace()
        for i,(x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda(), num_test, test_class, prototype)
            _, top_labels = pred_y.topk(1, dim=-1)
            acc_count += (top_labels.view(-1)==y.cuda()).sum().cpu().numpy()
            total_count += y.shape[0]
        acc = acc_count*1.0/total_count
        acc = acc.item()
        return acc

    @torch.no_grad()
    def inference(self,image, num_test, test_class, prototype):
        logits = self.model(image, num_test, test_class, prototype, test=True)
        return logits.float().softmax(dim=-1)
