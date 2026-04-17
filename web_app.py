"""
NOVA UNIVERSE TITAN MAX - Web Application
Полный код без заглушек
Запуск: python web_app.py
"""

import os
import sys
import json
import threading
import time
import re
from datetime import datetime
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# Добавляем путь к текущей директории для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем основной класс NOVA
from nova_titan_max import NovaTitanMax, USER_DATA_DIR, CONFIG

# Импортируем все модули из единого файла
from modules_all import (
    StreamGenerator, register_stream_endpoint,
    FlowEngine, AnalyticsPro, Vault, TranslatePro, PodcastGenerator,
    Crawler, Scheduler, DeepSearch, GameWorld,
    VoiceEngine, VisionEngine, VectorStore, SmartHomeMQTT,
    CalendarEngine, EmailEngine, TranslateEngine, PluginManager,
    SyncEngine, MonitorEngine, NotificationEngine, AnalyticsEngine, OnlineLearner
)

# ========== FLASK APP ==========
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ========== КОНФИГУРАЦИЯ АДМИНКИ ==========
ADMIN_SECRET_PATH = os.environ.get('ADMIN_SECRET_PATH', '/admin-d4f8e3a2')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'change-me')
MAINTENANCE_FLAG_FILE = os.path.join(USER_DATA_DIR, '.maintenance_mode')

def check_admin_auth():
    """Проверка мастер-пароля в заголовке"""
    provided = request.headers.get('X-Admin-Password')
    return provided == ADMIN_PASSWORD

def admin_route(rule, **options):
    """Декоратор для скрытых админских путей"""
    return app.route(ADMIN_SECRET_PATH + rule, **options)

def is_maintenance_mode():
    return os.path.exists(MAINTENANCE_FLAG_FILE)

@app.before_request
def check_maintenance():
    if request.path.startswith(ADMIN_SECRET_PATH):
        return None
    if request.path.startswith('/static') or request.path == '/health':
        return None
    if is_maintenance_mode():
        return render_template('maintenance.html'), 503

# ========== ИНИЦИАЛИЗАЦИЯ NOVA ==========
print("🚀 Загрузка NOVA TITAN MAX...")
nova = NovaTitanMax()
print("✅ Модель готова к работе")

# ========== ИНИЦИАЛИЗАЦИЯ МОДУЛЕЙ ==========
analytics_pro = AnalyticsPro(os.path.join(USER_DATA_DIR, 'analytics.db'))
vault = Vault(os.path.join(USER_DATA_DIR, 'vault'))
translate_pro = TranslatePro()
podcast_gen = PodcastGenerator(nova)
crawler = Crawler(nova.vectors)
scheduler = Scheduler()
deep_search = DeepSearch(nova.wiki, nova.vectors, None)
game_world = GameWorld(nova)
flow_engine = FlowEngine()

# Регистрируем стриминг
register_stream_endpoint(app, nova)

# Запускаем планировщик
scheduler.start()

# ========== ПОЛНОЦЕННЫЙ АГЕНТ ==========
class NovaAgent:
    """Автономный агент для выполнения цепочек задач"""
    def __init__(self, nova):
        self.nova = nova
        self.tools = {
            "calendar": self._calendar_tool,
            "wiki": self._wiki_tool,
            "web": self._web_tool,
            "files": self._file_tool,
            "email": self._email_tool,
            "reminder": self._reminder_tool,
        }

    def _calendar_tool(self, action: str, params: Dict) -> str:
        if action == "add":
            title = params.get("title")
            start_str = params.get("start")
            try:
                start = datetime.fromisoformat(start_str) if start_str else datetime.now()
            except:
                start = datetime.now()
            self.nova.calendar.add(title, start)
            return f"✅ Событие '{title}' добавлено в календарь"
        elif action == "list":
            events = self.nova.calendar.upcoming(5)
            if not events:
                return "📭 Нет предстоящих событий"
            return "📅 Предстоящие события:\n" + "\n".join(f"• {e['start'][:16]}: {e['title']}" for e in events)
        elif action == "delete":
            eid = params.get("id")
            if eid:
                self.nova.calendar.delete(eid)
                return f"✅ Событие удалено"
        return "❌ Неизвестное действие с календарём"

    def _wiki_tool(self, action: str, params: Dict) -> str:
        query = params.get("query", "")
        if not query:
            return "❌ Укажите запрос для поиска"
        result = self.nova.wiki.answer(query)
        return result if result else "Ничего не найдено"

    def _web_tool(self, action: str, params: Dict) -> str:
        query = params.get("query", "")
        if not query:
            return "❌ Укажите запрос для веб-поиска"
        if hasattr(self.nova, 'web') and self.nova.web:
            results = self.nova.web.search(query, 3)
            if results:
                return "🌐 Результаты:\n" + "\n".join(f"• {r.get('title', '')}: {r.get('snippet', '')[:100]}..." for r in results)
        return "🌐 Веб-поиск недоступен"

    def _file_tool(self, action: str, params: Dict) -> str:
        path = params.get("path", ".")
        if action == "list":
            try:
                files = os.listdir(path)
                if not files:
                    return "📭 Папка пуста"
                return "📁 Файлы:\n" + "\n".join(f"• {f}" for f in files[:20])
            except Exception as e:
                return f"❌ Ошибка чтения папки: {e}"
        elif action == "read":
            filepath = params.get("filepath")
            if not filepath:
                return "❌ Укажите путь к файлу"
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()[:2000]
                return f"📄 Содержимое {filepath}:\n{content}"
            except Exception as e:
                return f"❌ Ошибка чтения файла: {e}"
        return "❌ Неизвестное действие с файлами"

    def _email_tool(self, action: str, params: Dict) -> str:
        if action == "send":
            to = params.get("to")
            subject = params.get("subject", "Сообщение от NOVA")
            body = params.get("body", "")
            if not to:
                return "❌ Укажите получателя"
            if self.nova.email.send(to, subject, body):
                return f"✅ Письмо отправлено на {to}"
            return "❌ Не удалось отправить письмо"
        return "❌ Неизвестное действие с почтой"

    def _reminder_tool(self, action: str, params: Dict) -> str:
        if action == "set":
            text = params.get("text", "Напоминание")
            delay = params.get("delay", 60)
            try:
                delay = int(delay)
            except:
                delay = 60
            self.nova.notify.reminder(text, delay)
            return f"⏰ Напоминание установлено через {delay} секунд"
        return "❌ Неизвестное действие с напоминаниями"

    def execute(self, task_description: str) -> str:
        """Выполнение задачи с автоматическим выбором инструментов"""
        prompt = f"""Ты — автономный агент. Доступные инструменты: {', '.join(self.tools.keys())}.
Задача: {task_description}

Верни JSON-массив шагов. Каждый шаг: {{"tool": "...", "action": "...", "params": {{...}}}}.
Если задача не требует инструментов, верни пустой массив [].
JSON:"""

        response = self.nova.chat(prompt)
        try:
            # Извлекаем JSON из ответа
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                steps = json.loads(json_match.group())
                results = []
                for step in steps:
                    tool = step.get("tool")
                    action = step.get("action")
                    params = step.get("params", {})
                    if tool in self.tools:
                        result = self.tools[tool](action, params)
                        results.append(result)
                if results:
                    return "\n".join(results)
                return "✅ Задача выполнена (инструменты не потребовались)"
            return f"Не удалось распарсить план. Ответ модели: {response[:200]}..."
        except Exception as e:
            return f"❌ Ошибка выполнения: {e}"

# Создаём экземпляр агента
agent = NovaAgent(nova)

# ========== ПУБЛИЧНЫЕ API ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message'}), 400
    message = data['message']
    mode = data.get('mode', 'auto')
    response = nova.chat(message, mode=mode)
    return jsonify({'response': response})

@app.route('/api/wiki', methods=['POST'])
def wiki():
    data = request.get_json()
    query = data.get('query', '')
    result = nova.wiki.answer(query)
    return jsonify({'result': result})

@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify(nova.get_stats())

# ========== АГЕНТ (ПОЛНОЦЕННЫЙ) ==========
@app.route('/api/agent/task', methods=['POST'])
def agent_task():
    data = request.get_json()
    task = data.get('task', '')
    if not task:
        return jsonify({'error': 'No task provided'}), 400
    result = agent.execute(task)
    return jsonify({'result': result})

# ========== АДМИНСКИЕ API ==========
@admin_route('/train', methods=['POST'])
def admin_train():
    if not check_admin_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json() or {}
    epochs = data.get('epochs', 5)
    threading.Thread(target=nova.beta_trainer.train, args=(epochs,), daemon=True).start()
    return jsonify({'status': 'training started', 'epochs': epochs})

@admin_route('/maintenance/on', methods=['POST'])
def admin_maintenance_on():
    if not check_admin_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    with open(MAINTENANCE_FLAG_FILE, 'w') as f:
        f.write('1')
    return jsonify({'status': 'maintenance mode enabled'})

@admin_route('/maintenance/off', methods=['POST'])
def admin_maintenance_off():
    if not check_admin_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    if os.path.exists(MAINTENANCE_FLAG_FILE):
        os.remove(MAINTENANCE_FLAG_FILE)
    return jsonify({'status': 'maintenance mode disabled'})

@admin_route('/status', methods=['GET'])
def admin_status():
    if not check_admin_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(nova.get_stats())

@admin_route('/reload-plugins', methods=['POST'])
def admin_reload_plugins():
    if not check_admin_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    count = nova.plugins.reload()
    return jsonify({'status': 'ok', 'plugins_loaded': count})

@admin_route('/save-model', methods=['POST'])
def admin_save_model():
    if not check_admin_auth():
        return jsonify({'error': 'Unauthorized'}), 401
    nova.save_all()
    return jsonify({'status': 'model saved'})

# ========== МОДУЛЬНЫЕ API ==========
@app.route('/api/vault/unlock', methods=['POST'])
def vault_unlock():
    data = request.get_json()
    pwd = data.get('password')
    try:
        vault.unlock(pwd)
        return jsonify({'status': 'unlocked', 'secrets': vault.list_secrets()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/vault/store', methods=['POST'])
def vault_store():
    data = request.get_json()
    name = data.get('name')
    secret = data.get('secret')
    try:
        vault.store(name, secret)
        return jsonify({'status': 'stored'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/vault/retrieve', methods=['POST'])
def vault_retrieve():
    data = request.get_json()
    name = data.get('name')
    try:
        secret = vault.retrieve(name)
        return jsonify({'secret': secret})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/translate/document', methods=['POST'])
def translate_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    temp_path = os.path.join('/tmp', file.filename)
    file.save(temp_path)
    try:
        if file.filename.endswith('.pdf'):
            result = translate_pro.translate_pdf(temp_path)
        elif file.filename.endswith('.docx'):
            result = translate_pro.translate_docx(temp_path)
        else:
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
            result = translate_pro.translate_text(text)
        return jsonify({'translated': result})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/translate/text', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text', '')
    src = data.get('source', 'ru')
    tgt = data.get('target', 'en')
    result = translate_pro.translate_text(text, src, tgt)
    return jsonify({'translated': result})

@app.route('/api/podcast/generate', methods=['POST'])
def podcast_generate():
    data = request.get_json()
    topic = data.get('topic')
    result = podcast_gen.generate_full(topic)
    return jsonify(result)

@app.route('/api/crawler/start', methods=['POST'])
def crawler_start():
    data = request.get_json()
    url = data.get('url')
    max_pages = data.get('max_pages', 50)
    def run_crawl():
        crawler.crawl(url, max_pages)
    threading.Thread(target=run_crawl, daemon=True).start()
    return jsonify({'status': 'crawling started', 'url': url})

@app.route('/api/scheduler/jobs', methods=['GET'])
def scheduler_jobs():
    return jsonify(scheduler.list_jobs())

@app.route('/api/scheduler/add', methods=['POST'])
def scheduler_add():
    data = request.get_json()
    job_type = data.get('type')
    time_str = data.get('time')
    # Простейшее добавление
    if job_type == 'daily' and time_str:
        scheduler.add_daily(time_str, lambda: print(f"Scheduled task at {time_str}"))
        return jsonify({'status': 'job added'})
    return jsonify({'error': 'Invalid job'}), 400

@app.route('/api/search/deep', methods=['POST'])
def deep_search_endpoint():
    data = request.get_json()
    query = data.get('query')
    result = deep_search.search_all(query)
    return jsonify(result)

@app.route('/api/game/start', methods=['POST'])
def game_start():
    data = request.get_json()
    setting = data.get('setting')
    character = data.get('character')
    intro = game_world.start(setting, character)
    return jsonify({'intro': intro})

@app.route('/api/game/action', methods=['POST'])
def game_action():
    data = request.get_json()
    choice = data.get('choice')
    response = game_world.action(choice)
    return jsonify({'response': response})

@app.route('/api/game/save', methods=['POST'])
def game_save():
    data = request.get_json()
    name = data.get('name')
    game_world.save(name)
    return jsonify({'status': 'saved'})

@app.route('/api/game/load', methods=['POST'])
def game_load():
    data = request.get_json()
    name = data.get('name')
    ok = game_world.load(name)
    return jsonify({'loaded': ok})

@app.route('/api/game/list', methods=['GET'])
def game_list():
    saves = game_world.list_saves() if hasattr(game_world, 'list_saves') else []
    return jsonify({'saves': saves})

@app.route('/api/flow/execute', methods=['POST'])
def flow_execute():
    data = request.get_json()
    graph = data.get('graph', {})
    input_data = data.get('input')
    result = flow_engine.execute(graph, input_data, {'nova': nova, 'wiki': nova.wiki})
    return jsonify({'result': result})

@app.route('/api/voice/speak', methods=['POST'])
def voice_speak():
    data = request.get_json()
    text = data.get('text', '')
    nova.voice.speak(text)
    return jsonify({'status': 'spoken'})

@app.route('/api/vision/extract', methods=['POST'])
def vision_extract():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    temp_path = os.path.join('/tmp', file.filename)
    file.save(temp_path)
    try:
        text = nova.vision.extract_text(temp_path)
        return jsonify({'text': text})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ========== ЗАПУСК ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
