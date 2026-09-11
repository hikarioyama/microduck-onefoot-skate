#!/usr/bin/python3
"""Read-only training dashboard, deliberately without hardware telemetry."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import re
import time

BASE=Path(__file__).resolve().parent


def status():
    pointer=BASE/'onefoot-active.json'
    if not pointer.exists():
        return dict(iteration=None,log_age_s=None,dwell=None,success=None,
                    label='動的重心移動タスクへ切り替え中',capture_error=None,practice_dwell=None,practice_success=None)
    active=json.loads(pointer.read_text())
    logfile=BASE/active['log_file']
    with logfile.open('rb') as f:
        f.seek(max(0,logfile.stat().st_size-32768)); text=f.read().decode(errors='replace')
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    its=re.findall(r'Learning iteration\s+(\d+/\d+)',text)
    def value(name):
        matches=re.findall(re.escape(name)+r':\s+([-+\d.eE]+)',text)
        return float(matches[-1]) if matches else None
    fraction=value('Episode_Metrics/onefoot/launch_fraction')
    def launch_ratio(metric):
        numerator=value('Episode_Metrics/onefoot/'+metric)
        return numerator/fraction if numerator is not None and fraction and fraction>0 else None
    return dict(iteration=its[-1] if its else None,label=active['label'],
                log_age_s=round(time.time()-logfile.stat().st_mtime),
                dwell=launch_ratio('launch_best_dwell'),success=launch_ratio('launch_success'),
                practice_dwell=value('Episode_Metrics/onefoot/best_dwell'),
                practice_success=value('Episode_Metrics/onefoot/success'),
                capture_error=value('Episode_Metrics/onefoot/capture_error'))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/':
            payload=(BASE/'onefoot-dashboard.html').read_bytes()
            mime='text/html; charset=utf-8'; code=200
        elif self.path=='/status.json':
            try: payload=json.dumps(status(),allow_nan=False).encode(); code=200
            except Exception as e:
                print('Status failure:',type(e).__name__,flush=True)
                payload=b'{"error":"status unavailable"}'; code=503
            mime='application/json'
        else:
            self.send_error(404); return
        self.send_response(code); self.send_header('Content-Type',mime)
        self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(payload)))
        self.end_headers(); self.wfile.write(payload)


if __name__=='__main__':
    print('Dashboard: http://127.0.0.1:8087/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8087),Handler).serve_forever()
