import os
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

app = FastAPI(title="Minha API de Classificação de IA")

# --- CONFIGURAÇÃO DA API HUGGING FACE ---
API_URL = "https://api-inference.huggingface.co/models/MagnusFelintoMV/ClassificacaoEmail"

# Pega o token do ambiente (Render) ou usa None se não tiver
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class ComentarioInput(BaseModel):
    texto: str

def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# --- ROTA DA API (BACKEND) ---
@app.post("/analisar-comentario")
def analisar_comentario(dados: ComentarioInput):
    # CORREÇÃO AQUI: Não checamos mais 'classificador_ia', checamos o TOKEN
    if not HF_TOKEN:
         raise HTTPException(status_code=500, detail="Token do Hugging Face não configurado no Environment.")

    try:
        output = query_huggingface({"inputs": dados.texto})

        # Tratamento para quando o modelo está "acordando" (Cold Start)
        if isinstance(output, dict) and "error" in output:
             return {
                 "comentario_original": dados.texto, 
                 "classificacao": "Carregando...", 
                 "confianca": 0.0, 
                 "aviso": f"O modelo está inicializando: {output['error']}"
             }

        # A resposta vem como lista de listas [[{label:..., score:...}]]
        # Verificamos se a lista veio vazia para evitar erro de índice
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
            melhor_resultado = output[0][0]
            return {
                "comentario_original": dados.texto,
                "classificacao": melhor_resultado.get('label', 'Desconhecido'),
                "confianca": melhor_resultado.get('score', 0.0)
            }
        else:
            # Caso a resposta venha em formato inesperado
            return {
                "comentario_original": dados.texto,
                "classificacao": "Erro API",
                "confianca": 0.0,
                "detalhe": str(output)
            }

    except Exception as e:
        print(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- INTERFACE CHAT (FRONTEND) ---
@app.get("/", response_class=HTMLResponse)
def home():
    # COLE AQUI O HTML COMPLETO DA NOSSA CONVERSA ANTERIOR (AQUELE GRANDE)
    # Vou deixar apenas o esqueleto aqui, mas você deve manter o HTML que já tem.
    html_content = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Classificador de E-mail IA</title>
        <style>
            :root { --bg-color: #f3f4f6; --chat-bg: #ffffff; --primary: #2563eb; --text-main: #1f2937; --text-muted: #6b7280; --user-msg-bg: #eff6ff; --border-color: #e5e7eb; }
            body { font-family: 'Inter', -apple-system, sans-serif; background-color: var(--bg-color); margin: 0; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 20px; box-sizing: border-box; }
            .intro-section { text-align: center; max-width: 600px; margin-bottom: 25px; animation: fadeIn 0.5s ease; }
            .intro-section h1 { color: var(--text-main); font-size: 28px; margin-bottom: 10px; font-weight: 800; }
            .intro-section p { color: var(--text-muted); line-height: 1.6; margin-bottom: 20px; }
            .legend-container { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
            .legend-item { font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; align-items: center; gap: 6px; border: 1px solid transparent; }
            .dot-legend { width: 8px; height: 8px; border-radius: 50%; }
            .l-hate { color: #9b1c1c; border-color: #fde8e8; } .l-hate .dot-legend { background: #c81e1e; }
            .l-offensive { color: #9c4221; border-color: #feecdc; } .l-offensive .dot-legend { background: #dd6b20; }
            .l-spam { color: #854d0e; border-color: #fef08a; } .l-spam .dot-legend { background: #ca8a04; }
            .l-neither { color: #046c4e; border-color: #def7ec; } .l-neither .dot-legend { background: #31c48d; }
            .chat-container { width: 100%; max-width: 500px; background: var(--chat-bg); display: flex; flex-direction: column; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1); border-radius: 16px; overflow: hidden; height: 600px; max-height: 70vh; border: 1px solid var(--border-color); }
            .header { padding: 15px 20px; background: #fff; border-bottom: 1px solid var(--border-color); font-weight: 600; color: var(--text-main); display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
            .messages-area { flex: 1; padding: 20px; overflow-y: auto; background-color: #ffffff; background-image: radial-gradient(#e5e7eb 1px, transparent 1px); background-size: 24px 24px; display: flex; flex-direction: column; gap: 15px; }
            .message-wrapper { display: flex; flex-direction: column; align-items: flex-end; animation: slideUp 0.3s ease; }
            .message-bubble { background-color: var(--user-msg-bg); color: #1e3a8a; padding: 12px 16px; border-radius: 18px 18px 0 18px; font-size: 15px; max-width: 85%; word-wrap: break-word; line-height: 1.4; }
            .classification-tag { font-size: 11px; font-weight: 700; margin-top: 6px; padding: 4px 10px; border-radius: 12px; display: inline-flex; align-items: center; gap: 5px; letter-spacing: 0.3px; }
            .tag-hate { background: #fde8e8; color: #9b1c1c; } .tag-offensive { background: #fff7ed; color: #9a3412; } .tag-spam { background: #fefce8; color: #854d0e; } .tag-neither { background: #f0fdf4; color: #166534; } .tag-loading { background: #f3f4f6; color: #6b7280; }
            .input-area { padding: 15px; background: #fff; border-top: 1px solid var(--border-color); display: flex; gap: 10px; }
            input { flex: 1; padding: 12px 16px; border: 1px solid #d1d5db; border-radius: 24px; outline: none; font-size: 14px; transition: all 0.2s; }
            input:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }
            button { background-color: var(--primary); color: white; border: none; padding: 0 24px; border-radius: 24px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
            button:hover { background-color: #1d4ed8; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    </head>
    <body>
        <div class="intro-section">
            <h1>Classificador de E-mails com IA</h1>
            <p>Este é um chat inteligente projetado para analisar o conteúdo de e-mails em tempo real. Escreva uma mensagem abaixo e nossa Inteligência Artificial irá classificá-la automaticamente.</p>
            <div class="legend-container">
                <div class="legend-item l-hate"><span class="dot-legend"></span> Discurso de Ódio</div>
                <div class="legend-item l-offensive"><span class="dot-legend"></span> Ofensivo</div>
                <div class="legend-item l-spam"><span class="dot-legend"></span> Spam</div>
                <div class="legend-item l-neither"><span class="dot-legend"></span> Neutro / Seguro</div>
            </div>
            <p style="font-size: 13px; margin-top: 15px; color: #9ca3af;">👇 Teste agora digitando na caixa abaixo</p>
        </div>
        <div class="chat-container">
            <div class="header">Chat Monitorado <span style="font-size: 11px; background:#dbeafe; color:#1e40af; padding: 2px 8px; border-radius:10px;">IA Ativa</span></div>
            <div class="messages-area" id="messagesArea"><div style="text-align: center; color: #d1d5db; font-size: 13px; margin-top: auto; margin-bottom: auto;">O histórico aparecerá aqui...</div></div>
            <div class="input-area">
                <input type="text" id="msgInput" placeholder="Digite seu e-mail para testar..." autocomplete="off">
                <button id="sendBtn" onclick="enviarMensagem()">Enviar</button>
            </div>
        </div>
        <script>
            const input = document.getElementById('msgInput');
            const messagesArea = document.getElementById('messagesArea');
            let isFirstMessage = true;
            input.addEventListener("keypress", function(event) { if (event.key === "Enter") { event.preventDefault(); enviarMensagem(); } });
            async function enviarMensagem() {
                const texto = input.value.trim();
                if (!texto) return;
                if (isFirstMessage) { messagesArea.innerHTML = ''; isFirstMessage = false; }
                input.value = '';
                const msgId = Date.now();
                adicionarBalao(texto, msgId);
                try {
                    const response = await fetch('/analisar-comentario', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texto: texto }) });
                    const data = await response.json();
                    if (response.ok) { atualizarClassificacao(msgId, data.classificacao, data.confianca); } else { atualizarErro(msgId); }
                } catch (error) { console.error(error); atualizarErro(msgId); }
            }
            function adicionarBalao(texto, id) {
                const wrapper = document.createElement('div');
                wrapper.className = 'message-wrapper';
                wrapper.innerHTML = `<div class="message-bubble">${texto}</div><div id="tag-${id}" class="classification-tag tag-loading">Analisando...</div>`;
                messagesArea.appendChild(wrapper);
                messagesArea.scrollTop = messagesArea.scrollHeight;
            }
            function atualizarClassificacao(id, labelOriginal, score) {
                const tagElement = document.getElementById(`tag-${id}`);
                if (!tagElement) return;
                let info = { nome: "Desconhecido", css: "tag-spam" };
                const label = labelOriginal ? labelOriginal.toLowerCase() : "desconhecido";
                const porcentagem = (score * 100).toFixed(0) + '%';
                if (label.includes("hate") || label === "label_1") { info = { nome: "Discurso de Ódio", css: "tag-hate" }; }
                else if (label.includes("offensive") || label === "label_2") { info = { nome: "Ofensivo", css: "tag-offensive" }; }
                else if (label.includes("spam") || label === "label_3") { info = { nome: "Spam / Lixo", css: "tag-spam" }; }
                else if (label.includes("neither") || label === "label_0") { info = { nome: "Neutro / Seguro", css: "tag-neither" }; }
                else if (label.includes("carregando")) { info = { nome: "Iniciando IA...", css: "tag-loading" }; }
                tagElement.className = `classification-tag ${info.css}`;
                tagElement.innerHTML = `${info.nome} <span style="opacity:0.6; margin-left:4px">(${porcentagem})</span>`;
            }
            function atualizarErro(id) { const tagElement = document.getElementById(`tag-${id}`); if (tagElement) { tagElement.style.color = '#dc2626'; tagElement.innerText = "Erro ao classificar"; } }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

