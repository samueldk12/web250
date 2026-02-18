# Face Recognition System

Sistema de reconhecimento facial multi-modelo com suporte a ensemble e detecção de múltiplas faces.

## Funcionalidades

- **Múltiplos Modelos**: ArcFace, FaceNet, FaceNet-512, VGG-Face, OpenFace, DeepID, SFace, GhostFaceNet
- **Detecção Multi-Face**: Detecta e reconhece múltiplas pessoas em uma única imagem
- **Ensemble**: Combine múltiplos modelos para maior precisão
- **Upload em Lote**: Suporte a arquivos ZIP para registro e reconhecimento em massa
- **Upscaling Automático**: Melhora imagens de baixa resolução automaticamente
- **Interface Web**: Frontend intuitivo para todas as operações

## Arquitetura

```
web250/
├── docker-compose.yml          # Orquestração dos containers
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── download_models.py      # Download de modelos pré-treinados
│   └── app/
│       ├── main.py             # API FastAPI
│       ├── config.py           # Configurações
│       ├── models/             # Wrappers dos modelos
│       ├── services/           # Ensemble e upscaling
│       └── storage/            # Banco de dados de faces
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── public/                 # HTML, CSS, JS
└── data/
    ├── faces/                  # Imagens registradas
    └── embeddings.json         # Cache de embeddings
```

## Requisitos

- Docker e Docker Compose
- 4GB+ RAM (recomendado 8GB para múltiplos modelos)
- ~5GB de espaço em disco (para modelos pré-treinados)

## Instalação Rápida

```bash
# Clonar o repositório
git clone <repo-url>
cd web250

# Iniciar os containers (baixa modelos automaticamente)
docker-compose up --build -d

# Acessar a interface
# http://localhost:3000
```

## Download Manual dos Modelos Pré-Treinados

Se preferir baixar os modelos manualmente antes do build:

### DeepFace Models

| Modelo | Tamanho | Link Direto |
|--------|---------|-------------|
| ArcFace | 137 MB | [Download](https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5) |
| FaceNet | 92 MB | [Download](https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5) |
| FaceNet-512 | 95 MB | [Download](https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5) |
| VGG-Face | 580 MB | [Download](https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5) |
| OpenFace | 15 MB | [Download](https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5) |
| DeepID | 1.6 MB | [Download](https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5) |
| SFace | 39 MB | [Download](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) |
| GhostFaceNet | 17 MB | [Download](https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5) |

### Face Detection Models

| Detector | Tamanho | Link Direto |
|----------|---------|-------------|
| RetinaFace | 112 MB | [Download](https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5) |

### Image Upscaling Model (Real-ESRGAN)

| Modelo | Tamanho | Link Direto |
|--------|---------|-------------|
| Real-ESRGAN x4 | 64 MB | [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |

### Instalação Manual

```bash
# Criar diretórios
mkdir -p backend/.deepface/weights
mkdir -p backend/.esrgan/models

# Baixar modelos DeepFace
cd backend/.deepface/weights
wget https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5
wget https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
wget https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5
wget https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
wget https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5
wget https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5
wget https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx -O face_recognition_sface_2021dec.onnx
wget https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5 -O ghostfacenet_v1.h5

# Baixar modelo de upscaling
cd ../../.esrgan/models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

## API Endpoints

### Saúde e Modelos

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/health` | Status da API e modelos disponíveis |
| GET | `/api/models` | Lista detalhada dos modelos |
| GET | `/api/ensemble-methods` | Métodos de ensemble disponíveis |

### Gerenciamento de Faces

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/faces` | Lista todas as faces registradas |
| POST | `/api/faces/register` | Registra uma nova face |
| POST | `/api/faces/register-zip` | Registra faces de um arquivo ZIP |
| DELETE | `/api/faces/{id}` | Remove uma face registrada |
| GET | `/api/faces/{id}/image` | Retorna a imagem de uma face |

### Reconhecimento

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/faces/recognize` | Reconhece face(s) em uma imagem |
| POST | `/api/faces/recognize-zip` | Reconhece faces em múltiplas imagens (ZIP) |

## Uso da API

### Registrar uma Face

```bash
curl -X POST http://localhost:8000/api/faces/register \
  -F "name=João Silva" \
  -F "image=@foto.jpg" \
  -F "models=ArcFace,Facenet"
```

### Reconhecer Faces

```bash
curl -X POST http://localhost:8000/api/faces/recognize \
  -F "image=@foto_grupo.jpg" \
  -F "models=ArcFace" \
  -F "threshold=0.7"
```

### Upload em Lote (ZIP)

**Estrutura do ZIP para registro:**
```
pessoas.zip
├── joao/
│   ├── foto1.jpg
│   └── foto2.jpg
├── maria/
│   └── foto1.jpg
└── pedro/
    ├── img1.png
    └── img2.png
```

```bash
curl -X POST http://localhost:8000/api/faces/register-zip \
  -F "zipfile_upload=@pessoas.zip" \
  -F "models=ArcFace"
```

## Configurações

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `DATA_PATH` | `/app/data` | Caminho para dados persistentes |
| `DEEPFACE_HOME` | `/app/.deepface` | Caminho para modelos DeepFace |
| `DISTANCE_THRESHOLD` | `0.7` | Threshold de distância (0-1, menor = mais restritivo) |

### Ajuste de Threshold

- **0.4**: Muito restritivo, poucas correspondências
- **0.6**: Balanceado
- **0.7**: Recomendado para uso geral
- **0.8**: Mais permissivo, pode ter falsos positivos

## Upscaling Automático

O sistema automaticamente aplica upscaling em faces detectadas com resolução menor que 112x112 pixels usando Real-ESRGAN, melhorando significativamente a precisão do reconhecimento para:

- Fotos de grupo com pessoas distantes
- Imagens de câmeras de segurança
- Capturas de tela ou thumbnails

## Modelos de Reconhecimento

| Modelo | Dimensão | Velocidade | Precisão | Uso Recomendado |
|--------|----------|------------|----------|-----------------|
| ArcFace | 512 | Média | Alta | Uso geral |
| FaceNet | 128 | Rápida | Média | Tempo real |
| FaceNet-512 | 512 | Média | Alta | Alta precisão |
| VGG-Face | 4096 | Lenta | Média | Legado |
| OpenFace | 128 | Rápida | Média | Dispositivos limitados |
| DeepID | 160 | Rápida | Média | Experimental |
| SFace | 128 | Rápida | Média-Alta | Balanceado |
| GhostFaceNet | 512 | Rápida | Alta | Eficiência |

## Métodos de Ensemble

| Método | Descrição |
|--------|-----------|
| `average` | Média das distâncias de todos os modelos |
| `weighted` | Média ponderada pela precisão do modelo |
| `voting` | Maioria dos modelos decide o match |
| `min` | Usa a menor distância (mais confiante) |
| `max` | Usa a maior distância (mais conservador) |

## Troubleshooting

### Erro de Memória

Se receber warnings de memória, aumente os limites do Docker:

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Face Não Detectada

1. Verifique se a imagem tem boa iluminação
2. Certifique-se que a face está visível e não obstruída
3. Use imagens com faces maiores que 100x100 pixels
4. O upscaling automático ajuda com faces pequenas

### Baixa Confiança nas Correspondências

1. Registre múltiplas fotos da mesma pessoa
2. Use fotos com diferentes ângulos e iluminações
3. Aumente o threshold se necessário
4. Use ensemble de múltiplos modelos

## Desenvolvimento

### Rodar Localmente (sem Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
python download_models.py
uvicorn app.main:app --reload --port 8000

# Frontend (em outro terminal)
cd frontend/public
python -m http.server 3000
```

### Estrutura do Código

- `backend/app/models/`: Implementações dos modelos (DeepFace, InsightFace)
- `backend/app/services/`: Serviços (ensemble, upscaling)
- `backend/app/storage/`: Persistência (face_db.py)
- `frontend/public/`: Interface web (HTML, CSS, JS)

## Licença

Este projeto utiliza bibliotecas de código aberto:
- [DeepFace](https://github.com/serengil/deepface) - MIT License
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - BSD License
- [FastAPI](https://fastapi.tiangolo.com/) - MIT License

## Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request
