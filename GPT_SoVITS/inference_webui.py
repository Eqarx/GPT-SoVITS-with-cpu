import os,re
import pdb

gpt_path = os.environ.get(
    "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
)
sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share=eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True"))
is_half = False
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa, torch
from feature_extractor import cnhubert
cnhubert.cnhubert_base_path=cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto
import torch
i18n = I18nAuto()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device:{device}')
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

def trans32(i, name):
    try:
        if i.dtype == torch.float16:
            i = i.to(torch.float32)
            print(f'{name}.to(torch.f32)')
    except Exception:
        print(f'failed {name}.to(torch.f32)')
    return i

bert_model = trans32(bert_model, 'bert_model')
bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")

        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = trans32(res, 'res')
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            key = trans32(key, 'key')
            value = trans32(value, 'value')
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
try:
    ssl_model = ssl_model.to(torch.float32)
    print(f'ssl_model.to(torch.f32)')
except Exception:
    pass
ssl_model = ssl_model.to(device)


def get_slicer_name(s):
    # 删除“/”以及“/”前的所有内容
    s = re.sub(r'.*/', '', s)
    # 删除最后一个“-”以及最后一个“-”之后的所有内容
    s = re.sub(r'-[^-]*$', '', s)
    return s

def get_slicer_text(s):
    #仅保留最后一个“|”之后的内容
    s = re.search(r'[^|]*$', s).group()
    return s

def get_slicer_audio(s):
    # 第一步：分割以获取第一个“|”之前的内容
    first_part = s.split('|', 1)[0]
    # 第二步：使用正则表达式找到最后一个斜杠组合之后的内容
    match = re.search(r'(\\|/|\\\\|//)[^\\/]*$', first_part)
    result = first_part[match.start()+len(match.group(1)):]
    for _ in range(5):
        result = re.sub(r'\|.*', '', result)
    print(f"result={result}")
    return result

def read_slice_text(slicer_dir, slicer_name):
    slicer_path = os.path.join(slicer_dir, r'output\asr_opt\slicer_'+slicer_name+'.list')
    with open(slicer_path, 'r', encoding = 'utf-8') as f:
        line = f.readline().strip()
    slicer_text = get_slicer_text(line)
    return slicer_text

def read_slice_audio(slicer_dir, slicer_name1):
    print(f'slicer_name={slicer_name1}')
    slicer_path = os.path.join(slicer_dir, r'output\asr_opt\slicer_'+slicer_name1+'.list')
    try:
        with open(slicer_path, 'r', encoding = 'utf-8') as f:
            line = f.readline().strip()
    except FileNotFoundError:
        return ''
    slicer_audio = get_slicer_audio(str(line))
    print(f'slice_audio={slicer_audio}')
    slicer_audio_path = os.path.join(slicer_dir, r'output\slicer_'+slicer_name1+'\\'+slicer_audio)
    print(f'slicer_audio_path={slicer_audio_path}')
    if os.path.exists(slicer_audio_path):
        return slicer_audio_path
    else:
        return ''

def get_audio_text(GPT_path):
    slicer_name = get_slicer_name(str(GPT_path))
    print(f'text_slicer_name = {slicer_name}')
    slicer_dir = os.path.dirname(os.path.dirname(__file__))
    try:
        return read_slice_text(slicer_dir, slicer_name)
    except FileNotFoundError:
        print(f"can't find slicer_name = {slicer_name}")
    slicer_name1 = re.sub(r'-[^-]*$', '', slicer_name)
    try:
        return read_slice_text(slicer_dir, slicer_name1)
    except FileNotFoundError:
        print(f"can't find slicer_name1 = {slicer_name1}")
    slicer_name2 = re.sub(r'_[^_]*$', '', slicer_name)
    try:
        return read_slice_text(slicer_dir, slicer_name2)
    except FileNotFoundError:
        print(f"can't find slicer_name2 = {slicer_name2}")
    print(f'Finally cannot find slicer_path, default to empty string.')
    return ''

def get_audio_slicer(GPT_path):
    slicer_name = get_slicer_name(GPT_path)
    print(f'audio_slicer_name = {slicer_name},trying...')
    slicer_dir = os.path.dirname(os.path.dirname(__file__))
    a1 = read_slice_audio(slicer_dir, slicer_name)
    if a1 == '':
        print('a falied')
    else:
        return a1
    slicer_name1 = re.sub(r'-[^-]*$', '', slicer_name)
    print(f'slicer_name1={slicer_name1},trying...')
    a2 = read_slice_audio(slicer_dir, slicer_name1)
    if a2 == '':
        print('a1 falied')
    else:
        return a2
    slicer_name2 = re.sub(r'_[^_]*$', '', slicer_name)
    print(f'slicer_name2={slicer_name2},trying...')
    a3 =  read_slice_audio(slicer_dir, slicer_name2)
    if a3 == '':
        print('a2 falied')
    else:
        return a3
    print(f'Cannot find slicer_audio_path, default to None.')
    return None



def change_sovits_weights(sovits_path):
    global vq_model,hps
    dict_s2=torch.load(sovits_path,map_location="cpu")
    try:
        dict_s2 = dict_s2.to(torch.float32)
        print(f'dict_s2.to(torch.f32)')
    except Exception:
        pass
    hps=dict_s2["config"]
    hps = trans32(hps, 'hps')
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    del vq_model.enc_q
    vq_model = trans32(vq_model, 'vq_model')
    vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    print("调用了'change_sovits_weights'")
change_sovits_weights(sovits_path)

def change_gpt_weights(gpt_path):
    
    global hz,max_sec,t2s_model,config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    dict_s1 = trans32(dict_s1, 'dict_s1')
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model = trans32(t2s_model, 't2s_model')
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    print("调用了'change_gpt_weights',尝试更新音频与文本...")
    inp_ref=get_audio_slicer(gpt_path)
    prompt_text=get_audio_text(gpt_path)
    return inp_ref,prompt_text
change_gpt_weights(gpt_path)

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio = trans32(audio, 'audio')
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language={
    i18n("中文"):"zh",
    i18n("英文"):"en",
    i18n("日文"):"ja"
}


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language):
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype = np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        wav16k = trans32(wav16k, 'wav16k')
        zero_wav_torch = torch.from_numpy(zero_wav)
        wav16k = wav16k.to(device)
        zero_wav_torch = zero_wav_torch.to(device)
        wav16k=torch.cat([wav16k,zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    texts = text.split("\n")
    audio_opt = []
    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        if prompt_language == "zh":
            bert1 = get_bert_feature(norm_text1, word2ph1)
            bert1 = trans32(bert1, 'bert1')
            bert1.to(device)
        else:
            bert1 = torch.zeros(
                (1024, len(phones1)),
                dtype= torch.float32,
            ).to(device)
        if text_language == "zh":
            bert2 = get_bert_feature(norm_text2, word2ph2)
            bert2 = trans32(bert2, 'bert2')
            bert2.to(device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
        bert = torch.cat([bert1, bert2], 1)
        bert = trans32(bert, 'bert')

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        all_phoneme_len = trans32(all_phoneme_len, 'all_phoneme_len')
        prompt = prompt_semantic.unsqueeze(0).to(device)
        prompt = trans32(prompt, 'prompt')
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config["inference"]["top_k"],
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        refer = trans32(refer, 'refer')
        refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        audio = trans32(audio, 'audio')
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}  # 不考虑省略号


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 5))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return [inp]
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s。" % item for item in inp.strip("。").split("。")])

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names,key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names,key=custom_sort_key), "__type__": "update"}

pretrained_sovits_name="GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
os.makedirs(GPT_weight_root,exist_ok=True)
def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append("%s/%s"%(SoVITS_weight_root,name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s"%(GPT_weight_root,name))
    return SoVITS_names,GPT_names
SoVITS_names,GPT_names = get_weights_names()


with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    
    gr.Markdown(
        value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )
    with gr.Group():
        gr.Markdown(value=i18n("*请填写需要合成的目标文本"))
        with gr.Row():
            inp_ref = gr.Audio(label=i18n("请上传参考音频"), type="filepath", value=get_audio_slicer(gpt_path))
            prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value=get_audio_text(gpt_path))
            prompt_language = gr.Dropdown(
                label=i18n("参考音频的语种"),choices=[i18n("中文"),i18n("英文"),i18n("日文")],value=i18n("中文")
            )
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path,interactive=True)
            SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path,interactive=True)
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
            SoVITS_dropdown.change(change_sovits_weights,[SoVITS_dropdown],[])
            GPT_dropdown.change(fn=change_gpt_weights,inputs=[GPT_dropdown],outputs=[inp_ref, prompt_text])

        with gr.Row():
            text = gr.Textbox(label=i18n("需要合成的文本"), value="")
            text_language = gr.Dropdown(
                label=i18n("需要合成的语种"),choices=[i18n("中文"),i18n("英文"),i18n("日文")],value=i18n("中文")
            )
            inference_button = gr.Button(i18n("合成语音"), variant="primary")
            output = gr.Audio(label=i18n("输出的语音"))
        inference_button.click(
            get_tts_wav,
            [inp_ref, prompt_text, prompt_language, text, text_language],
            [output],
        )

        gr.Markdown(value=i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"))
        with gr.Row():
            text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"),value="")
            button1 = gr.Button(i18n("凑五句一切"), variant="primary")
            button2 = gr.Button(i18n("凑50字一切"), variant="primary")
            button3 = gr.Button(i18n("按中文句号。切"), variant="primary")
            text_opt = gr.Textbox(label=i18n("切分后文本"), value="")
            button1.click(cut1, [text_inp], [text_opt])
            button2.click(cut2, [text_inp], [text_opt])
            button3.click(cut3, [text_inp], [text_opt])
        gr.Markdown(value=i18n("后续将支持混合语种编码文本输入。"))

app.queue(concurrency_count=511, max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=True,
    server_port=infer_ttswebui,
    quiet=True,
)
