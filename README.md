# FlaskTtS
Simple Flask App for running Text-to-speech as a service

Currently supports the following TTS engines:
- [Style2TSS](https://github.com/yl4579/StyleTTS2)

## API Endpoints

### API Documentation

Once the service is running, you can access the Swagger UI documentation at:
`http://localhost:5001/api/docs`

### Text-to-Speech Operations

- `POST /tts/synthesize` - Create a new TTS job
- `GET /tts/jobs/{job_id}` - Get job status
- `GET /tts/jobs/{job_id}/download` - Download completed audio file
- `DELETE /tts/jobs/{job_id}` - Delete a specific job
- `GET /tts/jobs` - List all jobs
- `DELETE /tts/jobs` - Delete all jobs

### Usage Example

```python
import requests

# Create a new TTS job
response = requests.post('http://localhost:5001/tts/synthesize',
                        json={'text': 'Hello, world!'})
job_id = response.json()['job_id']

# Check job status
status = requests.get(f'http://localhost:5001/tts/jobs/{job_id}')
print(status.json())

# Download the audio file when completed
if status.json()['status'] == 'COMPLETED':
    audio = requests.get(f'http://localhost:5001/tts/jobs/{job_id}/download')
    with open('output.mp3', 'wb') as f:
        f.write(audio.content)
```

## Docker installation (recommended)

### Prerequisites
- Docker
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit installed

### Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FlaskTtS
```

2. Build and start the Docker container:
```bash
docker-compose up --build
```

3. The service will be available at `http://localhost:5001`

## Development

To run tests:
```bash
pytest
```

To run the web service locally:
```bash
pip install -r requirements.txt # create a virtual environment first if you want
FLASK_ENV=dev python run.py
```

To run the worker
```bash
huey_consumer.py huey_consumer.py flasktts.tasks.tasks.huey -w 1
```

## Configuration

The service can be configured through environment variables:
- `FLASK_ENV`: Set to 'development' for debug mode
- `PORT`: Server port (default: 5001)

## License

All my code is licensed under the MIT license. See the LICENSE file for more information.
Style2TSS is also permissively licensed but depends on `espeak-ng`, which is GPL-licensed.

## Roadmap (Todo)

1. Replace `espeak-ng` with [OpenPhonomizer](https://github.com/NeuralVox/OpenPhonemizer)
2. Add support for more TTS engines (e.g. Tortoise, Bark, etc.)
3. Give up on Huey and rewrite the whole thing with Celery?
4. CI/CD pipeline
5. Add more tests
6. SSE and/or websockets for job status updates

See companion project [Hoarder2Pod](https://github.com/DerekParks/Hoarder2Pod)
