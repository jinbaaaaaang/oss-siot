# -*- coding: utf-8 -*-
"""
Colab에서 FastAPI 서버를 실행하고 ngrok으로 터널링하는 스크립트

사용 방법:
1. Colab에서 이 파일을 업로드하거나 내용을 복사
2. 필요한 패키지 설치
3. 이 스크립트 실행
4. 생성된 ngrok URL을 프론트엔드에 설정
"""

import os
import sys
import subprocess
import threading
import time
import requests
from pathlib import Path

# ===== 설정 =====
# ngrok 토큰 (https://ngrok.com에서 무료 가입 후 발급)
NGROK_TOKEN = ""  # 여기에 ngrok 토큰 입력 (예: "2abc123def456ghi789jkl012mno345pq")

# 서버 포트
SERVER_PORT = 8000

# GPU 메모리 최적화 환경 변수 설정
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ===== 패키지 설치 =====
print("📦 필요한 패키지 설치 중...")
packages = [
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "pyngrok",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "torch",
    "scikit-learn",
    "numpy",
    "requests"
]

for package in packages:
    try:
        __import__(package.split("[")[0])
        print(f"  ✅ {package} 이미 설치됨")
    except ImportError:
        print(f"  📥 {package} 설치 중...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
        if result.returncode == 0:
            print(f"  ✅ {package} 설치 완료")
        else:
            print(f"  ⚠️ {package} 설치 실패 (코드: {result.returncode})")

# ===== ngrok 설정 =====
if NGROK_TOKEN:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_TOKEN)
    print("✅ ngrok 토큰 설정 완료")
    use_ngrok = True
else:
    print("⚠️ ngrok 토큰이 설정되지 않았습니다.")
    print("💡 ngrok 토큰 발급 방법:")
    print("   1. https://ngrok.com 접속")
    print("   2. 무료 회원가입")
    print("   3. Dashboard > Your Authtoken 복사")
    print("   4. 이 스크립트의 NGROK_TOKEN 변수에 붙여넣기")
    use_ngrok = False

# ===== 작업 디렉토리 설정 =====
# Colab에서 실행 시 /content 디렉토리 사용
if os.path.exists("/content"):
    base_dir = Path("/content")
    # 프로젝트가 클론되어 있는지 확인
    if (base_dir / "siot-OSS").exists():
        os.chdir(base_dir / "siot-OSS" / "backend")
        sys.path.insert(0, str(base_dir / "siot-OSS" / "backend"))
        print(f"✅ 작업 디렉토리: {os.getcwd()}")
    else:
        print("⚠️ siot-OSS 프로젝트를 찾을 수 없습니다.")
        print("💡 GitHub에서 클론하세요:")
        print("   !git clone https://github.com/your-username/siot-OSS.git")
        print("   %cd siot-OSS/backend")
else:
    # 로컬 실행 시
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    sys.path.insert(0, str(backend_dir))
    print(f"✅ 작업 디렉토리: {os.getcwd()}")

# ===== 서버 실행 함수 =====
def run_server():
    """FastAPI 서버 실행"""
    try:
        print(f"[서버] uvicorn 시작 중... (포트: {SERVER_PORT})")
        print(f"[서버] 작업 디렉토리: {os.getcwd()}")
        print(f"[서버] Python 경로: {sys.executable}")
        
        # app.main:app이 존재하는지 확인
        if not os.path.exists('app/main.py'):
            print(f"❌ [서버] app/main.py 파일을 찾을 수 없습니다!")
            print(f"   현재 디렉토리: {os.getcwd()}")
            print(f"   파일 목록: {os.listdir('.')}")
            return
        
        # 서버 실행 (에러 출력 포함)
        result = subprocess.run([
            sys.executable, '-m', 'uvicorn',
            'app.main:app',
            '--host', '0.0.0.0',
            '--port', str(SERVER_PORT)
            # --reload 제거 (Colab에서는 불필요하고 문제를 일으킬 수 있음)
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # 서버가 종료된 경우 출력
        if result.returncode != 0:
            print(f"❌ [서버] 서버가 종료되었습니다 (코드: {result.returncode})")
            print(f"[서버] 출력:\n{result.stdout}")
    except Exception as e:
        print(f"❌ [서버] 서버 실행 오류: {e}")
        import traceback
        traceback.print_exc()

# ===== 기존 프로세스 정리 =====
print("\n" + "="*80)
print("🧹 기존 프로세스 정리 중...")
print("="*80)

# 기존 uvicorn 프로세스 종료
try:
    result = subprocess.run(['pkill', '-f', 'uvicorn'], capture_output=True, check=False)
    if result.returncode == 0:
        print("✅ 기존 uvicorn 프로세스 종료됨")
        time.sleep(2)  # 프로세스 종료 대기
    else:
        print("ℹ️ 실행 중인 uvicorn 프로세스 없음")
except:
    pass

# 포트 해제 시도
try:
    result = subprocess.run(['fuser', '-k', f'{SERVER_PORT}/tcp'], capture_output=True, check=False)
    time.sleep(1)
except:
    pass

# GPU 메모리 정리 (가능한 경우)
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU 메모리 캐시 정리됨")
except:
    pass

# ===== 서버 시작 =====
print("\n" + "="*80)
print("🚀 FastAPI 서버 시작 중...")
print("="*80)

# 백그라운드에서 서버 시작
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# 서버 시작 대기
print("⏳ 서버 시작 대기 중... (모델 로딩 포함 약 2-5분 소요)")
server_ready = False
max_wait = 60  # 최대 10분 대기

for i in range(max_wait):
    try:
        response = requests.get(f'http://localhost:{SERVER_PORT}/health', timeout=5)
        if response.status_code == 200:
            print(f"\n✅ 서버 준비 완료! ({i+1}번째 시도, 약 {i*10}초)")
            server_ready = True
            break
    except requests.exceptions.ConnectionError:
        # 아직 서버가 시작 중 - 진행 상황 출력
        if i % 3 == 0:  # 30초마다 출력
            elapsed = (i + 1) * 10
            print(f"   ⏳ 대기 중... ({elapsed}초 경과, {i+1}/{max_wait} 시도)")
    except Exception as e:
        # 기타 오류 발생 시 출력
        if i % 3 == 0:  # 30초마다 출력
            elapsed = (i + 1) * 10
            print(f"   ⏳ 대기 중... ({elapsed}초 경과, 오류: {type(e).__name__})")
    
    time.sleep(10)  # 10초마다 체크

if not server_ready:
    print("\n⚠️ 서버가 시작되지 않았습니다.")
    print("\n🔍 디버깅 정보:")
    print("="*80)
    
    # 서버 프로세스 확인
    try:
        import subprocess as sp
        result = sp.run(['ps', 'aux'], capture_output=True, text=True)
        if 'uvicorn' in result.stdout:
            print("✅ uvicorn 프로세스가 실행 중입니다")
            print(result.stdout)
        else:
            print("❌ uvicorn 프로세스가 실행되지 않았습니다")
    except:
        print("⚠️ 프로세스 확인 실패")
    
    # 포트 확인
    try:
        result = sp.run(['netstat', '-tuln'], capture_output=True, text=True)
        if str(SERVER_PORT) in result.stdout:
            print(f"✅ 포트 {SERVER_PORT}이 사용 중입니다")
        else:
            print(f"❌ 포트 {SERVER_PORT}이 열려있지 않습니다")
    except:
        print("⚠️ 포트 확인 실패")
    
    # 파일 확인
    print(f"\n📁 현재 디렉토리: {os.getcwd()}")
    print(f"📁 파일 목록:")
    try:
        files = os.listdir('.')
        for f in files[:10]:  # 처음 10개만
            print(f"   - {f}")
        if len(files) > 10:
            print(f"   ... 외 {len(files)-10}개 파일")
    except:
        print("   파일 목록을 가져올 수 없습니다")
    
    # app/main.py 확인
    if os.path.exists('app/main.py'):
        print("✅ app/main.py 파일 존재")
    else:
        print("❌ app/main.py 파일이 없습니다!")
        if os.path.exists('app'):
            print("   app 디렉토리 내용:")
            try:
                for f in os.listdir('app'):
                    print(f"     - {f}")
            except:
                pass
    
    print("\n💡 해결 방법:")
    print("   1. 서버를 포그라운드로 실행하여 에러 메시지 확인:")
    print("      subprocess.run([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'])")
    print("   2. 모델 로딩이 오래 걸릴 수 있습니다 (5-10분). 잠시 후 다시 시도하세요")
    print("   3. GPU가 활성화되어 있는지 확인: !nvidia-smi")
    print("   4. 필요한 패키지가 모두 설치되었는지 확인")
else:
    # ===== ngrok 터널 생성 =====
    if use_ngrok:
        print("\n" + "="*80)
        print("🌐 ngrok 터널 생성 중...")
        print("="*80)
        
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(SERVER_PORT)
            # NgrokTunnel 객체에서 URL 추출
            if hasattr(tunnel, 'public_url'):
                public_url = tunnel.public_url
            elif hasattr(tunnel, 'url'):
                public_url = tunnel.url
            else:
                public_url = str(tunnel).replace('NgrokTunnel: "', '').split('"')[0]
            
            print(f"\n✅ 터널 생성 완료!")
            print(f"\n📝 공개 URL: {public_url}")
            print(f"\n💡 프론트엔드 설정:")
            print(f"   .env 파일에 다음을 추가하세요:")
            print(f"   VITE_COLAB_API_URL={public_url}")
            print(f"\n   (프론트엔드 코드에서 자동으로 /api/poem/generate를 추가합니다)")
            print(f"\n⚠️ 주의:")
            print(f"   - Colab 세션이 종료되면 URL이 변경됩니다")
            print(f"   - 무료 ngrok은 세션당 2시간 제한이 있습니다")
            print(f"   - URL은 이 셀을 다시 실행하면 변경됩니다")
        except Exception as e:
            print(f"❌ ngrok 터널 생성 실패: {e}")
            print(f"\n💡 대안:")
            print(f"   1. Colab의 '코드 셀' > '변수' 메뉴에서 포트 포워딩 사용")
            print(f"   2. 또는 로컬에서 ngrok 사용: ngrok http 8000")
    else:
        print(f"\n📝 로컬 서버 주소:")
        print(f"   http://localhost:{SERVER_PORT}")
        print(f"\n💡 Colab에서 외부 접근을 위해서는 ngrok이 필요합니다.")

# ===== 서버 상태 확인 =====
print("\n" + "="*80)
print("📊 서버 상태")
print("="*80)

try:
    response = requests.get(f'http://localhost:{SERVER_PORT}/health', timeout=10)
    if response.status_code == 200:
        health_data = response.json()
        print(f"✅ 서버 정상 작동 중")
        print(f"   - 모델 타입: {health_data.get('model_type', 'N/A')}")
        print(f"   - 디바이스: {health_data.get('device', 'N/A')}")
        print(f"   - GPU 사용: {health_data.get('has_gpu', False)}")
    else:
        print(f"⚠️ 서버 응답 오류: {response.status_code}")
except Exception as e:
    print(f"⚠️ 서버 상태 확인 실패: {e}")

print("\n" + "="*80)
print("✅ 설정 완료!")
print("="*80)
print("\n💡 서버를 계속 실행하려면 이 셀을 실행 상태로 유지하세요.")
print("💡 서버를 중지하려면 런타임 > 세션 중단을 선택하세요.")

