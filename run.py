import subprocess
import threading
import time
import webbrowser
import os

def run_backend():
    os.chdir('backend')
    subprocess.run(['python', '-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', '8000'])

def run_frontend():
    time.sleep(3)
    webbrowser.open('http://localhost:8501')
    os.chdir('frontend')
    subprocess.run(['streamlit', 'run', 'app.py', '--server.port', '8501'])

if __name__ == "__main__":
    print("Starting Rice Disease Detection System...")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:8501")
    
    backend_thread = threading.Thread(target=run_backend)
    frontend_thread = threading.Thread(target=run_frontend)
    
    backend_thread.start()
    frontend_thread.start()
    
    backend_thread.join()
    frontend_thread.join()
