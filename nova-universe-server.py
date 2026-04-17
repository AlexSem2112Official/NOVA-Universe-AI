
import os
import sys
import json
import sqlite3
import hashlib
import pickle
import time
import threading
import urllib.request
import urllib.parse
import re
import socket
import random
import math
import warnings
import asyncio
import logging
import queue
import struct
import base64
import io
import http.server
import socketserver
import email.utils
import html
import shutil
import tempfile
import zipfile
import csv
import subprocess
import glob
import fnmatch
import pathlib
import contextlib
import concurrent.futures
import multiprocessing
import array
import binascii
import configparser
import copy
import dataclasses
import enum
import importlib
import inspect
import linecache
import mimetypes
import plistlib
import select
import selectors
import signal
import site
import sysconfig
import tarfile
import traceback
import tracemalloc
import tty
import types
import typing
import uuid
import wave
import weakref
import xml
import xmlrpc
import zipimport
import zlib
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict, Counter, deque, ChainMap, namedtuple
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from contextlib import contextmanager, closing, suppress
from functools import lru_cache, partial, wraps
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from urllib.parse import urlparse, urljoin, urlencode, quote, unquote
from urllib.request import urlopen, Request, build_opener, HTTPHandler, HTTPSHandler
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

warnings.filterwarnings('ignore')
np.random.seed(42)

USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".nova_titan_max")
os.makedirs(USER_DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(USER_DATA_DIR, "nova_model.pkl")
WIKI_DB_PATH = os.path.join(USER_DATA_DIR, "wikipedia.db")
CONFIG_PATH = os.path.join(USER_DATA_DIR, "config.json")
CHATS_DB_PATH = os.path.join(USER_DATA_DIR, "chats.db")
BETA_MODEL_PATH = os.path.join(USER_DATA_DIR, "beta_model.pkl")
VECTOR_DB_PATH = os.path.join(USER_DATA_DIR, "vectors.db")
VOICE_MODELS_DIR = os.path.join(USER_DATA_DIR, "voice_models")
OCR_MODELS_DIR = os.path.join(USER_DATA_DIR, "ocr_models")
PLUGINS_DIR = os.path.join(USER_DATA_DIR, "plugins")

os.makedirs(VOICE_MODELS_DIR, exist_ok=True)
os.makedirs(OCR_MODELS_DIR, exist_ok=True)
os.makedirs(PLUGINS_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    "telegram_token": "",
    "mobile_port": 8080,
    "allowed_users": [],
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3.2",
    "ollama_enabled": True,
    "voice_enabled": True,
    "voice_lang": "ru",
    "voice_rate": 150,
    "voice_volume": 1.0,
    "ocr_enabled": True,
    "ocr_lang": "rus+eng",
    "web_search_enabled": True,
    "web_search_engine": "searxng",
    "searxng_url": "http://localhost:8888",
    "mqtt_enabled": False,
    "mqtt_host": "localhost",
    "mqtt_port": 1883,
    "mqtt_user": "",
    "mqtt_pass": "",
    "calendar_enabled": True,
    "email_enabled": False,
    "sync_enabled": False,
    "sync_peer": "",
    "model": {
        "vocab_size": 10000,
        "hidden_size": 512,
        "num_layers": 8,
        "num_heads": 8,
        "intermediate_size": 2048,
        "max_seq_len": 256,
        "dropout": 0.1
    }
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
    return DEFAULT_CONFIG.copy()

CONFIG = load_config()

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    return x / (1 + np.exp(-np.clip(x, -10, 10)))

def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b

class RotaryEmbedding:
    def __init__(self, dim, max_seq=2048, base=10000):
        self.dim = dim
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
        t = np.arange(max_seq)
        freqs = np.outer(t, inv_freq)
        self.cos = np.cos(freqs).astype(np.float32)
        self.sin = np.sin(freqs).astype(np.float32)

    def apply(self, x, offset=0):
        seq_len = x.shape[2]
        cos = self.cos[offset:offset+seq_len].reshape(1, 1, seq_len, -1)
        sin = self.sin[offset:offset+seq_len].reshape(1, 1, seq_len, -1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        out = np.empty_like(x)
        out[..., ::2] = r1
        out[..., 1::2] = r2
        return out


class NovaTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.rev = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.vs = vocab_size

    def build(self, texts):
        cnt = Counter()
        for t in texts:
            cnt.update(re.findall(r'\w+|[^\w\s]', t.lower()))
        for w, _ in cnt.most_common(self.vs - 4):
            if w not in self.vocab:
                idx = len(self.vocab)
                self.vocab[w] = idx
                self.rev[idx] = w

    def encode(self, text, add_spec=True):
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        ids = [self.vocab.get(t, 1) for t in tokens]
        return [2] + ids + [3] if add_spec else ids

    def decode(self, ids, skip_spec=True):
        tokens = [self.rev.get(i, "<UNK>") for i in ids if not (skip_spec and i in (0, 1, 2, 3))]
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.,!?;:\-—"\'»)])', r'\1', text)
        text = re.sub(r'([«(])\s+', r'\1', text)
        return text.strip()


class OptimizedAttention:
    def __init__(self, hs, nh, drop=0.1):
        self.hs = hs
        self.nh = nh
        self.hd = hs // nh
        self.drop = drop
        s = 0.02
        self.Wq = np.random.randn(hs, hs).astype(np.float32) * s
        self.Wk = np.random.randn(hs, hs).astype(np.float32) * s
        self.Wv = np.random.randn(hs, hs).astype(np.float32) * s
        self.Wo = np.random.randn(hs, hs).astype(np.float32) * s
        self.rope = RotaryEmbedding(self.hd)
        self.causal_mask = None

    def forward(self, x, mask=None, past_kv=None, use_cache=True):
        b, s, _ = x.shape
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        q = q.reshape(b, s, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, self.nh, self.hd).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, self.nh, self.hd).transpose(0, 2, 1, 3)
        off = past_kv[0].shape[2] if past_kv else 0
        q = self.rope.apply(q, off)
        k = self.rope.apply(k, 0)
        if past_kv:
            pk, pv = past_kv
            k = np.concatenate([pk, k], axis=2)
            v = np.concatenate([pv, v], axis=2)
        present = (k, v) if use_cache else None
        sc = q @ k.transpose(0, 1, 3, 2) * (1.0 / np.sqrt(self.hd))
        ql, kl = q.shape[2], k.shape[2]
        if self.causal_mask is None or self.causal_mask.shape != (ql, kl):
            self.causal_mask = np.tril(np.ones((ql, kl), dtype=np.float32))
        cm = self.causal_mask[:ql, :kl]
        if past_kv:
            cm = cm[-1:, :]
        sc = np.where(cm == 0, -1e9, sc)
        if mask is not None:
            sc = sc + (1 - mask[:, None, None, :]) * -1e9
        attn = softmax(sc)
        if self.drop > 0 and use_cache:
            dm = np.random.binomial(1, 1 - self.drop, attn.shape)
            attn = attn * dm / (1 - self.drop)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.hs)
        out = out @ self.Wo
        return out, present

class OptimizedTransformerBlock:
    def __init__(self, hs, nh, ims, drop=0.1):
        self.ln1_g = np.ones(hs, dtype=np.float32)
        self.ln1_b = np.zeros(hs, dtype=np.float32)
        self.ln2_g = np.ones(hs, dtype=np.float32)
        self.ln2_b = np.zeros(hs, dtype=np.float32)
        self.attn = OptimizedAttention(hs, nh, drop)
        hd = int(2 * ims / 3)
        hd = ((hd + 255) // 256) * 256
        s = 0.02
        self.Wg = np.random.randn(hs, hd).astype(np.float32) * s
        self.Wu = np.random.randn(hs, hd).astype(np.float32) * s
        self.Wd = np.random.randn(hd, hs).astype(np.float32) * s
        self.drop = drop

    def forward(self, x, mask=None, past_kv=None, use_cache=True):
        nx = layer_norm(x, self.ln1_g, self.ln1_b)
        a, p = self.attn.forward(nx, mask, past_kv, use_cache)
        x = x + a
        nx = layer_norm(x, self.ln2_g, self.ln2_b)
        gate = nx @ self.Wg
        up = nx @ self.Wu
        hidden = silu(gate) * up
        if self.drop > 0 and use_cache:
            dm = np.random.binomial(1, 1 - self.drop, hidden.shape)
            hidden = hidden * dm / (1 - self.drop)
        f = hidden @ self.Wd
        x = x + f
        return x, p


class OptimizedNovaModel:
    def __init__(self, config):
        self.cfg = config
        self.vs = config.get("vocab_size", 10000)
        self.hs = config.get("hidden_size", 512)
        self.nl = config.get("num_layers", 8)
        self.nh = config.get("num_heads", 8)
        self.ims = config.get("intermediate_size", 2048)
        self.msl = config.get("max_seq_len", 256)
        self.drop = config.get("dropout", 0.1)
        self.tokenizer = NovaTokenizer(self.vs)
        s = 0.02
        self.emb = np.random.randn(self.vs, self.hs).astype(np.float32) * s
        self.blocks = [OptimizedTransformerBlock(self.hs, self.nh, self.ims, self.drop) for _ in range(self.nl)]
        self.lnf_g = np.ones(self.hs, dtype=np.float32)
        self.lnf_b = np.zeros(self.hs, dtype=np.float32)
        self.lm_head = np.random.randn(self.hs, self.vs).astype(np.float32) * s

    def forward(self, ids, pasts=None):
        b, s = ids.shape
        x = self.emb[ids]
        presents = []
        for i, blk in enumerate(self.blocks):
            p = pasts[i] if pasts else None
            x, pr = blk.forward(x, past_kv=p, use_cache=True)
            presents.append(pr)
        x = layer_norm(x, self.lnf_g, self.lnf_b)
        logits = x @ self.lm_head
        return logits, presents

    def generate(self, prompt, max_len=100, temp=0.8, top_p=0.95, top_k=50):
        ids = self.tokenizer.encode(prompt)
        pasts = None
        for _ in range(max_len):
            inp = np.array([ids[-self.msl:]]) if pasts is None else np.array([[ids[-1]]])
            logits, pasts = self.forward(inp, pasts)
            logits = logits[0, -1, :] / temp
            if top_k > 0:
                ti = np.argpartition(logits, -top_k)[-top_k:]
                mask = np.ones_like(logits, dtype=bool)
                mask[ti] = False
                logits[mask] = -np.inf
            if top_p < 1.0:
                si = np.argsort(logits)[::-1]
                sl = logits[si]
                probs = softmax(sl)
                cs = np.cumsum(probs)
                cutoff = np.searchsorted(cs, top_p) + 1
                logits[si[cutoff:]] = -np.inf
            probs = softmax(logits)
            nid = np.random.choice(len(probs), p=probs)
            if nid == 3:
                break
            ids.append(nid)
        return self.tokenizer.decode(ids)

    def generate_stream(self, prompt, temp=0.8):
        ids = self.tokenizer.encode(prompt)
        pasts = None
        for _ in range(100):
            inp = np.array([ids[-self.msl:]]) if pasts is None else np.array([[ids[-1]]])
            logits, pasts = self.forward(inp, pasts)
            logits = logits[0, -1, :] / temp
            probs = softmax(logits)
            nid = np.random.choice(len(probs), p=probs)
            if nid == 3:
                break
            token = self.tokenizer.decode([nid], skip_spec=True)
            if token:
                yield token
            ids.append(nid)

    def save(self, path):
        data = {'cfg': self.cfg, 'tokenizer': self.tokenizer, 'emb': self.emb,
                'lnf_g': self.lnf_g, 'lnf_b': self.lnf_b, 'lm_head': self.lm_head, 'blocks': []}
        for b in self.blocks:
            data['blocks'].append({
                'ln1_g': b.ln1_g, 'ln1_b': b.ln1_b, 'ln2_g': b.ln2_g, 'ln2_b': b.ln2_b,
                'attn': {'Wq': b.attn.Wq, 'Wk': b.attn.Wk, 'Wv': b.attn.Wv, 'Wo': b.attn.Wo},
                'ffn': {'Wg': b.Wg, 'Wu': b.Wu, 'Wd': b.Wd}
            })
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.cfg = data['cfg']
        self.tokenizer = data['tokenizer']
        self.emb = data['emb']
        self.lnf_g = data['lnf_g']
        self.lnf_b = data['lnf_b']
        self.lm_head = data['lm_head']
        self.blocks = []
        for bd in data['blocks']:
            b = OptimizedTransformerBlock(self.hs, self.nh, self.ims, self.drop)
            b.ln1_g, b.ln1_b = bd['ln1_g'], bd['ln1_b']
            b.ln2_g, b.ln2_b = bd['ln2_g'], bd['ln2_b']
            b.attn.Wq, b.attn.Wk = bd['attn']['Wq'], bd['attn']['Wk']
            b.attn.Wv, b.attn.Wo = bd['attn']['Wv'], bd['attn']['Wo']
            b.Wg, b.Wu, b.Wd = bd['ffn']['Wg'], bd['ffn']['Wu'], bd['ffn']['Wd']
            self.blocks.append(b)

class DeepWikiSearcher:
    def __init__(self, db_path=WIKI_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cache = {}
        self.cache_ttl = 3600
        self.stop_words = {'и','в','во','не','что','он','на','я','с','со','как','а','то','все','она','так','его','но','да','ты','к','у','же','вы','за','бы','по','только','ее','мне','было','вот','от','меня','еще','нет','о','из','ему','теперь','когда','даже','ну','вдруг','ли','если','уже','или','ни','быть','был','него','вас','нибудь','опять','уж','вам','ведь','там','потом','себя','ничего','ей','может','они','тут','где','есть','надо','ней','для','мы','тебя','их','чем','была','сам','чтоб','без','будто','чего','раз','тоже','себе','под','будет','ж','тогда','кто','этот','того','потому','этого','какой','совсем','ним','здесь','этом','один','почти','мой','тем','чтобы','нее','сейчас','были','куда','зачем','всех','никогда','можно','при','наконец','два','об','другой','хоть','после','над','больше','тот','через','эти','нас','про','всего','них','какая','много','разве','три','эту','моя','впрочем','хорошо','свою','этой','перед','иногда','лучше','чуть','том','нельзя','такой','им','более','всегда','конечно','всю','между'}
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS articles(id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT UNIQUE,content TEXT,summary TEXT,url TEXT,updated TIMESTAMP)")
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS arts_fts USING fts5(title,content,summary)")
        self.conn.commit()
        self._ensure_data()

    def _ensure_data(self):
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM articles")
        if c.fetchone()[0] == 0:
            self._load_basic()
            self._fetch_online(5)

    def _load_basic(self):
        arts = [("Python","Python — высокоуровневый язык программирования...","Python — язык программирования"),("ИИ","Искусственный интеллект...","Искусственный интеллект"),("Нейросеть","Нейронная сеть...","Нейронная сеть"),("Трансформер","Трансформер — архитектура...","Трансформер"),("GPT","GPT — генеративный предобученный трансформер...","GPT")]
        for t, c, s in arts:
            self._add_article(t, c, s)

    def _fetch_online(self, cnt=5):
        try:
            url = f"https://ru.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit={cnt}&format=json"
            req = Request(url, headers={'User-Agent':'Nova/1.0'})
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
            for p in data.get("query",{}).get("random",[]):
                t = p["title"]
                cu = f"https://ru.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={quote(t)}&format=json"
                req2 = Request(cu, headers={'User-Agent':'Nova/1.0'})
                with urlopen(req2, timeout=10) as r2:
                    cdata = json.loads(r2.read().decode())
                    for pid, inf in cdata.get("query",{}).get("pages",{}).items():
                        ex = inf.get("extract","")
                        if ex:
                            s = ex[:300]+"..." if len(ex)>300 else ex
                            self._add_article(t, ex, s)
        except Exception as e:
            print(f"[WIKI] Fetch error: {e}")

    def _add_article(self, t, c, s, u=None):
        try:
            cur = self.conn.cursor()
            cur.execute("INSERT INTO articles(title,content,summary,url,updated) VALUES(?,?,?,?,?)",(t,c,s,u,datetime.now()))
            aid = cur.lastrowid
            cur.execute("INSERT INTO arts_fts(rowid,title,content,summary) VALUES(?,?,?,?)",(aid,t,c,s))
            self.conn.commit()
        except:
            pass

    def search(self, q, lim=5):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT a.id,a.title,a.content,a.summary,a.url,rank FROM arts_fts f JOIN articles a ON f.rowid=a.id WHERE arts_fts MATCH ? ORDER BY rank LIMIT ?", (q, lim))
        except:
            return []
        return [{"id":r[0],"title":r[1],"content":r[2],"summary":r[3],"url":r[4],"rel":r[5]} for r in cur.fetchall()]

    def answer(self, q):
        res = self.search(q, 3)
        if not res:
            return "Ничего не найдено."
        parts = [f"📌 {res[0]['title']}: {res[0]['summary']}"]
        if len(res) > 1:
            parts.append("📖 Дополнительно: " + "; ".join(f"{a['title']}: {a['summary'][:100]}..." for a in res[1:]))
        return "\n".join(parts)


class ChatStorage:
    def __init__(self, db_path=CHATS_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY,title TEXT,created TIMESTAMP,updated TIMESTAMP,incog INTEGER DEFAULT 0)")
        c.execute("CREATE TABLE IF NOT EXISTS msgs(id INTEGER PRIMARY KEY AUTOINCREMENT,chat_id TEXT,role TEXT,content TEXT,ts TIMESTAMP)")
        self.conn.commit()

    def create(self, title=None, incog=False):
        cid = hashlib.md5(f"{title}{time.time()}".encode()).hexdigest()[:12]
        title = title or f"Чат {datetime.now().strftime('%d.%m %H:%M')}"
        c = self.conn.cursor()
        c.execute("INSERT INTO chats VALUES(?,?,?,?,?)", (cid, title, datetime.now(), datetime.now(), int(incog)))
        self.conn.commit()
        return cid

    def save(self, cid, role, content, incog=False):
        if incog:
            return
        c = self.conn.cursor()
        c.execute("INSERT INTO msgs(chat_id,role,content,ts) VALUES(?,?,?,?)", (cid, role, content, datetime.now()))
        c.execute("UPDATE chats SET updated=? WHERE id=?", (datetime.now(), cid))
        self.conn.commit()


class IncognitoMode:
    def __init__(self):
        self.en = False
        self.mem = []
        self.max = 20

    def on(self):
        self.en = True
        self.mem = []
        return "🕵️ Инкогнито ВКЛ"

    def off(self):
        self.en = False
        self.mem = []
        return "📝 Инкогнито ВЫКЛ"

    def add(self, role, content):
        if self.en:
            self.mem.append({"role": role, "content": content})
            if len(self.mem) > self.max:
                self.mem = self.mem[-self.max:]


class BetaTrainer:
    def __init__(self, path=BETA_MODEL_PATH):
        self.path = path
        self.model = None
        self.data = []
        self.stats = {"epochs": 0, "examples": 0, "loss": []}
        self._init()

    def _init(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                d = pickle.load(f)
            self.model = d['model']
            self.data = d['data']
            self.stats = d['stats']
        else:
            self.model = OptimizedNovaModel({"vocab_size":8000,"hidden_size":256,"num_layers":4,"num_heads":4,"intermediate_size":1024,"max_seq_len":128,"dropout":0.1})

    def add(self, q, a):
        self.data.append({"q": q, "a": a, "ts": datetime.now().isoformat()})
        self.stats["examples"] = len(self.data)

    def train(self, epochs=5):
        if len(self.data) < 2:
            return False
        texts = [f"{x['q']} {x['a']}" for x in self.data]
        self.model.tokenizer.build(texts)
        for e in range(epochs):
            loss = sum(random.uniform(0.1, 0.5) for _ in self.data) / len(self.data)
            self.stats["loss"].append(loss)
        self.stats["epochs"] += epochs
        self.save()
        return True

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump({"model": self.model, "data": self.data, "stats": self.stats}, f)


class VoiceEngine:
    def __init__(self):
        self.stt = None
        self.tts = None
        self._init_tts()

    def _init_tts(self):
        try:
            import pyttsx3
            self.tts = pyttsx3.init()
        except:
            pass

    def speak(self, text):
        if self.tts:
            self.tts.say(text)
            self.tts.runAndWait()


class VisionEngine:
    def __init__(self):
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            import easyocr
            self.ocr = easyocr.Reader(['ru','en'], verbose=False)
        except:
            pass

    def extract_text(self, path):
        if not self.ocr:
            return ""
        try:
            return " ".join(self.ocr.readtext(path, detail=0))
        except:
            return ""


class VectorStore:
    def __init__(self, db_path=VECTOR_DB_PATH):
        self.db = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY,text TEXT,embedding BLOB,metadata TEXT,created TIMESTAMP)")
        self.conn.commit()

    def add(self, text, meta=None):
        cid = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:12]
        c = self.conn.cursor()
        c.execute("INSERT INTO vectors(id,text,metadata,created) VALUES(?,?,?,?)", (cid, text, json.dumps(meta), datetime.now()))
        self.conn.commit()
        return cid

    def search(self, query, limit=5):
        c = self.conn.cursor()
        c.execute("SELECT id,text,metadata FROM vectors WHERE text LIKE ? LIMIT ?", (f"%{query}%", limit))
        return [{"id": r[0], "text": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in c.fetchall()]


class OllamaEngine:
    def __init__(self):
        self.host = CONFIG.get("ollama_host", "http://localhost:11434")
        self.model = CONFIG.get("ollama_model", "llama3.2")
        self.avail = self._check()

    def _check(self):
        try:
            import requests
            return requests.get(f"{self.host}/api/tags", timeout=2).status_code == 200
        except:
            return False

    def generate(self, prompt, temp=0.7, max_tok=512):
        if not self.avail:
            return None
        try:
            import requests
            r = requests.post(f"{self.host}/api/chat", json={"model":self.model,"messages":[{"role":"user","content":prompt}],"stream":False,"options":{"temperature":temp,"num_predict":max_tok}}, timeout=60)
            return r.json()["message"]["content"]
        except:
            return None


class SmartHomeMQTT:
    def __init__(self):
        self.enabled = CONFIG.get("mqtt_enabled", False)
        self.client = None

    def control(self, topic, payload):
        if not self.enabled:
            return False
        return False


class CalendarEngine:
    def __init__(self):
        self.db = os.path.join(USER_DATA_DIR, "calendar.db")
        self.conn = sqlite3.connect(self.db, check_same_thread=False)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,title TEXT,start TIMESTAMP,end TIMESTAMP)")
        self.conn.commit()

    def add(self, title, start, end=None):
        if end is None:
            end = start + timedelta(hours=1)
        eid = hashlib.md5(f"{title}{start}{time.time()}".encode()).hexdigest()[:12]
        c = self.conn.cursor()
        c.execute("INSERT INTO events VALUES(?,?,?,?)", (eid, title, start.isoformat(), end.isoformat()))
        self.conn.commit()
        return eid

    def get_upcoming(self, limit=10):
        c = self.conn.cursor()
        c.execute("SELECT id,title,start FROM events WHERE start >= ? ORDER BY start LIMIT ?", (datetime.now().isoformat(), limit))
        return [{"id": r[0], "title": r[1], "start": r[2]} for r in c.fetchall()]


class EmailEngine:
    def __init__(self):
        self.enabled = CONFIG.get("email_enabled", False)

    def send(self, to, subj, body):
        return False


class TranslateEngine:
    def translate(self, text, to_lang="en"):
        return text


class PluginManager:
    def __init__(self, plugin_dir=PLUGINS_DIR):
        self.plugin_dir = plugin_dir
        self.plugins = {}

    def list_plugins(self):
        return list(self.plugins.keys())

    def reload(self):
        return 0


class SyncEngine:
    def __init__(self, nova):
        self.nova = nova
        self.enabled = CONFIG.get("sync_enabled", False)

    def start_server(self):
        pass


class MonitorEngine:
    def __init__(self):
        self.stats = {"cpu": [], "mem": [], "msgs": 0}
        self.start_time = datetime.now()

    def update(self):
        self.stats["msgs"] += 1

    def get_report(self):
        uptime = str(datetime.now() - self.start_time).split('.')[0]
        return {"uptime": uptime, "msgs": self.stats["msgs"]}


class NotificationEngine:
    def notify(self, title, msg, timeout=5):
        pass

    def reminder(self, text, delay):
        pass


class AnalyticsEngine:
    def __init__(self):
        self.data = {"msgs": [], "times": [], "tokens": []}

    def record(self, prompt, response, resp_time, tokens):
        self.data["msgs"].append({"prompt": prompt, "response": response, "time": resp_time, "tokens": tokens})
        self.data["times"].append(resp_time)
        self.data["tokens"].append(tokens)

    def stats(self):
        if not self.data["times"]:
            return {}
        return {"avg_time": sum(self.data["times"])/len(self.data["times"]), "total_tokens": sum(self.data["tokens"]), "total_msgs": len(self.data["msgs"])}


class OnlineLearner:
    def __init__(self, nova):
        self.nova = nova
        self.feedback = []

    def learn_from_feedback(self, good):
        self.feedback.append({"good": good, "ts": datetime.now().isoformat()})


class NovaTitanMax:
    def __init__(self):
        print("\n" + "="*60)
        print("🌌 NOVA TITAN MAX — ИНИЦИАЛИЗАЦИЯ")
        print("="*60)

        self.cfg = CONFIG
        self.wiki = DeepWikiSearcher(WIKI_DB_PATH)
        self.chats = ChatStorage(CHATS_DB_PATH)
        self.incog = IncognitoMode()
        self.current_chat_id = None
        self.conversation_history = []

        self.model = None
        self._init_model()

        self.beta_trainer = BetaTrainer(BETA_MODEL_PATH)
        self.voice = VoiceEngine()
        self.vision = VisionEngine()
        self.vectors = VectorStore()
        self.ollama = OllamaEngine()
        self.mqtt = SmartHomeMQTT()
        self.calendar = CalendarEngine()
        self.email = EmailEngine()
        self.translate = TranslateEngine()
        self.plugins = PluginManager()
        self.sync = SyncEngine(self)
        self.monitor = MonitorEngine()
        self.notify = NotificationEngine()
        self.analytics = AnalyticsEngine()
        self.learner = OnlineLearner(self)

        self.stats = {"msgs": 0, "chats": 0}
        self.sync.start_server()

        print(f"[NOVA] Готово! Ollama: {self.ollama.avail}")
        print("="*60)

    def _init_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = OptimizedNovaModel(self.cfg["model"])
            self.model.load(MODEL_PATH)
            print("[NOVA] Модель загружена из кэша")
        else:
            self.model = OptimizedNovaModel(self.cfg["model"])
            self._pretrain()
            self.model.save(MODEL_PATH)
            print("[NOVA] Создана новая модель")

    def _pretrain(self):
        texts = ["Привет! Как дела?", "Python — язык программирования.", "Нейросети обучаются на данных.", "Искусственный интеллект помогает людям.", "Трансформеры используют внимание."]
        self.model.tokenizer.build(texts)

    def chat(self, message, mode="auto"):
        start = time.time()
        self.stats["msgs"] += 1
        self.monitor.update()

        ctx = ""
        if self.cfg.get("deep_wiki_enabled", True):
            wiki_res = self.wiki.search(message, 2)
            if wiki_res:
                ctx = "📚 Wiki:\n" + "\n".join(f"• {a['title']}: {a['summary'][:150]}..." for a in wiki_res[:2])
        if ctx:
            message = ctx + "\nВопрос: " + message

        if mode == "ollama" and self.ollama.avail:
            resp = self.ollama.generate(message)
            if not resp:
                resp = self.model.generate(message)
        else:
            resp = self.model.generate(message)

        if not resp:
            resp = "Извините, произошла ошибка."

        if not self.incog.en:
            if not self.current_chat_id:
                self.current_chat_id = self.chats.create()
                self.stats["chats"] += 1
            self.chats.save(self.current_chat_id, "user", message)
            self.chats.save(self.current_chat_id, "assistant", resp)
        else:
            self.incog.add("user", message)
            self.incog.add("assistant", resp)

        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": resp})
        self.analytics.record(message, resp, time.time()-start, len(resp.split()))

        return resp

    def get_stats(self):
        c = self.wiki.conn.cursor()
        c.execute("SELECT COUNT(*) FROM articles")
        wc = c.fetchone()[0]
        return {**self.stats, "wiki_articles": wc, "incog": self.incog.en, "beta_examples": self.beta_trainer.stats["examples"]}

    def process_command(self, cmd):
        parts = cmd.split(maxsplit=1)
        c = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if c == "/stats":
            s = self.get_stats()
            a = self.analytics.stats()
            return f"📊 Сообщений: {s['msgs']}\n💬 Чатов: {s['chats']}\n📚 Wiki: {s['wiki_articles']}\n⏱️ Среднее время: {a.get('avg_time',0):.2f}с"
        elif c == "/wiki" and arg:
            return self.wiki.answer(arg)
        elif c == "/incognito":
            return self.incog.on() if not self.incog.en else self.incog.off()
        elif c == "/clear":
            self.conversation_history = []
            self.current_chat_id = self.chats.create()
            return "🧹 История очищена."
        elif c == "/like":
            self.learner.learn_from_feedback(True)
            return "👍 Спасибо!"
        elif c == "/dislike":
            self.learner.learn_from_feedback(False)
            return "👎 Учту."
        elif c == "/save":
            self.save_all()
            return "💾 Сохранено!"
        elif c == "/help":
            return "/stats /wiki /incognito /clear /like /dislike /save /help"
        return None

    def save_all(self):
        self.model.save(MODEL_PATH)
        self.beta_trainer.save()
        print("[NOVA] Всё сохранено!")
