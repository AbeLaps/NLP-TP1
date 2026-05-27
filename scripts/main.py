from openai import OpenAI
import os
import pandas as pd
import argparse
from dotenv import load_dotenv
import time
import json
import re
import math
from itertools import combinations
from collections import Counter

load_dotenv()  # carrega o .env 


# Argumento de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Modelo a ser usado")
args = parser.parse_args()

client = OpenAI(

    api_key=os.environ.get("GROQ_API_KEY"),

    base_url="https://api.groq.com/openai/v1",

)


systemMessage = 'Você é um assistente de escrita que produz rascunhos de resposta com diferentes tons, por exemplo: formal; curto e objetivo; cordial; mais empático; mais direto, entre outros.'

testConfigs = ["baseline", "soft", "aggressive"]

messages = pd.read_json('../data/messages.json')

# Função principal para requisição 
def requestLLMResponse(prompt, model): 
    try:
        resposta = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemMessage},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return resposta.choices[0].message.content
    except Exception as e:
        print(f"Erro: {e}")


def customizePrompt(passage, config='baseline'):
    if config == 'baseline':
        prompt = (
            f'produzia 5 respostas para a seguinte mensagem: {passage}, com tons diferentes.\n'
            'Formato de resposta (obrigatório), uma lista JSON neste formato exato, os valores de probabilidade devem ser ignorados e apenas copiados para o JSON de resposta final\n'
            '[\n'
            '{{"tom": "*", "probabilidade": 100, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": 100, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": 100, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": 100, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": 100, "resposta": "texto aqui"}}\n'
            ']'
        )
    elif config == 'soft':
        prompt = (
            f'Dado a seguinte mensagem: {passage}\n'
            'Gere 5 respostas com tons diferentes.\n'
            'Para cada um, indique o tom e uma probabilidade estimada de ser a resposta mais adequada (em %).\n'
            'Formato de resposta (obrigatório), uma lista JSON neste formato exato:\n'
            '[\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}}\n'
            ']'
        )
    elif config == 'aggressive':
        prompt = (
            f'Dado a seguinte mensagem: {passage}\n'
            'Gere 5 rascunhos de resposta MUITO diferentes entre si.\n'
            'Evite repetir palavras, estruturas de frase ou abordagens entre as respostas.\n'
            'Varie o comprimento, a abertura e o estilo de cada resposta.\n'
            'Para cada um, indique o tom, a probabilidade estimada (%) e quando essa resposta seria ideal.\n'
            'Formato de resposta (obrigatório), uma lista JSON neste formato exato:\n'
            '[\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}}\n'
            ']'
        )
    else:
        raise ValueError('As configurações devem ser baseline, soft ou aggressive')

    return prompt

def parseRespostas(raw):
    """Extrai lista de respostas do JSON retornado pelo modelo."""
    try:
        raw = raw.replace('{{', '{').replace('}}', '}')
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []



# Metricas

def tokenize(text):
    return re.findall(r'\w+', text.lower())
 
 
def jaccard(texts):
    """Média do índice de Jaccard entre todos os pares de respostas."""
    scores = []
    for a, b in combinations(texts, 2):
        set_a = set(tokenize(a))
        set_b = set(tokenize(b))
        if not set_a and not set_b:
            scores.append(1.0)
        elif not set_a or not set_b:
            scores.append(0.0)
        else:
            scores.append(len(set_a & set_b) / len(set_a | set_b))
    return round(sum(scores) / len(scores), 4) if scores else None
 
 
def perplexity(texts):
    """
    Pseudo-perplexidade baseada em unigrama do corpus das respostas.
    Valores menores = vocabulário mais concentrado/repetitivo.
    """
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize(t))
 
    if not all_tokens:
        return None
 
    freq = Counter(all_tokens)
    total = len(all_tokens)
    probs = [freq[t] / total for t in all_tokens]
    log_prob = sum(math.log(p) for p in probs)
    return round(math.exp(-log_prob / total), 4)
 
 
def selfBleu(texts):
    """
    Self-BLEU simplificado: média do BLEU-1 de cada resposta
    usando as demais como referência.
    Valores maiores = respostas mais similares entre si.
    """
    def bleu1(hypothesis, references):
        hyp_tokens = tokenize(hypothesis)
        if not hyp_tokens:
            return 0.0
        ref_tokens = set()
        for r in references:
            ref_tokens.update(tokenize(r))
        matches = sum(1 for t in hyp_tokens if t in ref_tokens)
        return matches / len(hyp_tokens)
 
    scores = []
    for i, hyp in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        scores.append(bleu1(hyp, refs))
 
    return round(sum(scores) / len(scores), 4) if scores else None
 
 
def distinctN(texts, n=2):
    """
    Distinct-N: proporção de n-gramas únicos em relação ao total.
    Valores maiores = maior variedade lexical.
    """
    all_ngrams = []
    for text in texts:
        tokens = tokenize(text)
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
 
    if not all_ngrams:
        return None
 
    return round(len(set(all_ngrams)) / len(all_ngrams), 4)


def calcularMetricas(respostas):
    """Recebe lista de dicts com chave 'resposta' e retorna métricas."""
    texts = [r.get('resposta', '') for r in respostas if r.get('resposta')]
    if len(texts) < 2:
        return {}
 
    return {
        'jaccard_medio':   jaccard(texts),
        'perplexidade':    perplexity(texts),
        'self_bleu':       selfBleu(texts),
        'distinct_1':      distinctN(texts, n=1),
        'distinct_2':      distinctN(texts, n=2),
    }



def salvarRespostas(model, config, msg_index, mensagem, respostas, metricas):
    """Salva um CSV por modelo/config com respostas e métricas."""
    model_slug = model.replace('/', '_')
    filename = f"resultados_{model_slug}_{config}.csv"
 
    rows = []
    for r in respostas:
        row = {
            'msg_index':      msg_index,
            'mensagem':       mensagem,
            'tom':            r.get('tom', ''),
            'resposta':       r.get('resposta', ''),
            'probabilidade':  r.get('probabilidade', ''),
            'ideal_quando':   r.get('ideal_quando', ''),
            'jaccard_medio':  metricas.get('jaccard_medio', ''),
            'perplexidade':   metricas.get('perplexidade', ''),
            'self_bleu':      metricas.get('self_bleu', ''),
            'distinct_1':     metricas.get('distinct_1', ''),
            'distinct_2':     metricas.get('distinct_2', ''),
        }
        rows.append(row)
 
    df = pd.DataFrame(rows)
 
    if os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)

# MAIN
for config in testConfigs:
    for i in range(len(messages) // 19):
        print('=' * 60)
        print(f'Modelo: {args.model}')
        print(f'Configuracao: {config}')
        print(f'Mensagem [{i}]: {messages["mensagem"][i][:60]}...')
 
        raw = requestLLMResponse(customizePrompt(messages['mensagem'][i], config), args.model)
        print(raw)
 
        respostas = parseRespostas(raw)
        if respostas:
            metricas = calcularMetricas(respostas)
            print(f'Metricas: {metricas}')
            salvarRespostas(
                model=args.model,
                config=config,
                msg_index=i,
                mensagem=messages['mensagem'][i],
                respostas=respostas,
                metricas=metricas,
            )
        else:
            print('Aviso: nao foi possivel parsear as respostas.')
 
        print('=' * 60)
        time.sleep(10)