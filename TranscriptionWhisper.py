import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import torch.nn.functional as F
import librosa
import warnings

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
        
        # Força o modelo a gerar em português 
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="portuguese", 
            task="transcribe"
        )

    def transcrever_audio(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> dict:
        """
        Recebe o batch de áudio (Mock do VAD) e retorna texto e confiança.
        """
        # Processamento da entrada para o formato do modelo
        input_features = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features.to(self.device, dtype=torch.float16)

        # Geração de texto com extração de probabilidades 
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decodificação dos tokens gerados para string
        transcricao = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

        # Cálculo da confiança 
        confianca = self._calcular_confianca(outputs.scores)

        return {
            "texto": transcricao,
            "confianca": confianca
        }

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

if __name__ == "__main__":
    transcritor = TranscritorWhisper()

    # Substitui pelo nome do teu ficheiro de teste
    caminho_audio = "audio6.ogg" 
    
    try:
        # Carrega o áudio já forçando a amostragem exigida pelo modelo
        audio_real, sr = librosa.load(caminho_audio, sr=16000)
        
        print(f"A processar o ficheiro: {caminho_audio}...")
        resultado = transcritor.transcrever_audio(audio_real)
        
        print("\n--- Saída do Componente I5 ---")
        print(f"Texto: {resultado['texto']}")
        print(f"Confiança: {resultado['confianca']}")
        
    except FileNotFoundError:
        print(f"Erro: O ficheiro '{caminho_audio}' não foi encontrado.")
        print("Grava um ficheiro rápido e coloca-o na mesma pasta do script.")