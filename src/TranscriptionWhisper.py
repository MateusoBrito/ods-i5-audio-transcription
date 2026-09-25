import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import torch.nn.functional as F
import librosa
import warnings
import re
import unicodedata
import spacy
import json
import datetime

warnings.filterwarnings("ignore", category=UserWarning)

class TranscritorWhisper:
    def __init__(self, model_name="openai/whisper-small", device="cuda"):
        """
        Inicializa o modelo e o processador. 
        """
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Configuração padrão de idioma para português
        self.language = "portuguese"
        self.task = "transcribe"

        try:
            self.nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
        except OSError:
            print("Aviso: Modelo do spaCy não encontrado. Execute: python -m spacy download pt_core_news_sm")
            self.nlp = None

    def transcrever_audio(self, audio_array: np.ndarray, sampling_rate: int = 16000, origem: str = "P5 pendente", timestamp_inicio: str = None) -> str:
        """
        Recebe o batch de áudio e metadados, retornando o payload JSON estruturado.
        """

        if not timestamp_inicio:
            # Mock para testes locais caso o VAD não envie o timestamp
            inicio_dt = datetime.datetime.now(datetime.timezone.utc)
        else:
            # Converte a string ISO 8601 recebida do P5/VAD para objeto datetime
            inicio_dt = datetime.datetime.fromisoformat(timestamp_inicio.replace('Z', '+00:00'))

        # Calcula a duração do áudio e encontra o timestamp final
        duracao_segundos = len(audio_array) / sampling_rate
        fim_dt = inicio_dt + datetime.timedelta(seconds=duracao_segundos)

        # Processamento da entrada para o formato do modelo
        input_features = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features.to(self.device, dtype=torch.float16)

        # Geração de texto forçando estritamente o idioma português
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                language=self.language,
                task=self.task,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decodificação dos tokens gerados para string
        transcricao = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        transcricao_limpa = self._preprocessar_texto(transcricao)
        # Cálculo da confiança 
        confianca = self._calcular_confianca(outputs.scores)
        duvida_explicita = bool(confianca < 0.5)

        payload = {
            "origem": origem,
            "timestamp_inicio": inicio_dt.isoformat(),
            "timestamp_fim": fim_dt.isoformat(),
            "texto_transcrito": transcricao,
            "texto_preprocessado": transcricao_limpa,
            "confianca_geral": confianca,
            "duvida_explicita": duvida_explicita
        }

        envelope = {
            "message_type": "event",
            "schema": "ods.varejo.transcricao",
            "schema_version": "1.0",
            "producer": "I5",
            "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "payload": payload
        }

        return json.dumps(envelope, ensure_ascii=False, indent=2)

    def _calcular_confianca(self, scores: tuple) -> float:
        """
        Converte as pontuações brutas (logits) em uma probabilidade média (0.0 a 1.0).
        """
        if not scores:
            return 0.0

        probabilidades_tokens = []
        
        # Itera sobre cada passo de geração (cada token gerado)
        for step_logits in scores:
            # step_logits tem shape (batch_size, vocab_size). 
            # Como estamos processando um áudio por vez, pegamos o índice 0.
            logits = step_logits[0]
            
            # Aplica Softmax para transformar logits em probabilidades (0 a 1)
            probs = F.softmax(logits, dim=-1)
            
            # Pega o valor da maior probabilidade (o token escolhido)
            prob_maxima = torch.max(probs).item()
            
            probabilidades_tokens.append(prob_maxima)
            
        # Calcula a média aritmética das probabilidades da frase
        confianca_media = sum(probabilidades_tokens) / len(probabilidades_tokens)
        
        # Arredonda para 4 casas decimais para manter o JSON limpo
        return round(confianca_media, 4)

    def _preprocessar_texto(self, texto: str) -> str:
        """
        Aplica limpeza leve no texto
        """
        texto = texto.lower()
        
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                        if unicodedata.category(c) != 'Mn')
        
        texto = re.sub(r'(.)\1{2,}', r'\1', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        if self.nlp:
            doc = self.nlp(texto)
            texto = " ".join([token.lemma_ for token in doc])
        
        return texto

if __name__ == "__main__":
    transcritor = TranscritorWhisper()

    # Substitui pelo nome do teu ficheiro de teste
    caminho_audio = "data/audio6.ogg" 
    
    try:
        audio_real, sr = librosa.load(caminho_audio, sr=16000)
        
        print(f"A processar o ficheiro: {caminho_audio}...\n")
        
        # Passando um timestamp e origem simulados para o teste local
        json_saida = transcritor.transcrever_audio(
            audio_array=audio_real, 
            origem="mic_balcao_1",
            timestamp_inicio="2026-09-16T14:05:00.000Z"
        )
        
        print("--- Saída do Componente I5 (Contrato JSON) ---")
        print(json_saida)
        
    except FileNotFoundError:
        print(f"Erro: O ficheiro '{caminho_audio}' não foi encontrado.")
        print("Grava um ficheiro rápido e coloca-o na mesma pasta do script.")