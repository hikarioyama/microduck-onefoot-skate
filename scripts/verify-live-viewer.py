"""Browser acceptance test on isolated copied checkpoints; never touches training."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
inputs=json.loads((ROOT/'local/live-viewer-test-inputs.json').read_text())
root=Path(inputs['root']);first=Path(inputs['first']);second=Path(inputs['second'])
state_file=ROOT/'local/live-viewer-test-state.json'
status_file=ROOT/'local/live-viewer-test-status.json'

def wait_status(predicate,seconds=60):
    deadline=time.time()+seconds;last={}
    while time.time()<deadline:
        if status_file.exists():last=json.loads(status_file.read_text())
        if predicate(last):return last
        time.sleep(.25)
    raise AssertionError(last)

def publish_state(stage):
    temp=state_file.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(run_root=str(root),run_dir=str(root/stage),stage=stage,recipe='continuous')))
    temp.replace(state_file)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

initial=wait_status(lambda s:s.get('status')=='running' and s.get('stage')=='balance-100')
assert initial['checkpoint']==str(root/'balance-100'/first.name)
# Locate a Playwright-managed Chromium without pinning one machine's cache path.
_candidates=sorted(Path.home().joinpath('.cache/ms-playwright').glob('chromium-*/**/chrome'))
if not _candidates:
    raise SystemExit('No Playwright Chromium found; run `playwright install chromium` first')
exe=_candidates[-1]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=str(exe),headless=True,
        args=['--disable-gpu','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':1550,'height':1000});errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8088/',wait_until='domcontentloaded')
    page.get_by_role('tab',name='Checkpoints').click(timeout=30000)
    def wait_label(text,seconds=35):
        deadline=time.time()+seconds
        while time.time()<deadline:
            tab=page.get_by_role('tab',name='Checkpoints')
            if tab.count() and tab.is_visible():tab.click()
            values=page.locator('input').evaluate_all('(xs)=>xs.map(x=>x.value)')
            if any(text in value for value in values):return
            page.wait_for_timeout(300)
        raise AssertionError(page.locator('body').inner_text())
    wait_label('balance-100/'+first.name)
    assert page.locator('canvas').count()>0
    assert page.locator('iframe').count()==0
    # A newly published real checkpoint must load without Use Latest or page reload.
    destination=root/'balance-100'/second.name
    shutil.copyfile(second,destination)
    updated=wait_status(lambda s:s.get('status')=='running' and s.get('checkpoint')==str(destination))
    wait_label('balance-100/'+second.name)
    assert updated['load_count']>initial['load_count'] and updated['sha256']==digest(second)
    # A bad same-name rewrite must not corrupt the current actor or roll it back.
    destination.write_bytes(b'incomplete checkpoint')
    failed=wait_status(lambda s:s.get('status')=='checkpoint_load_failed')
    time.sleep(3)
    failed=json.loads(status_file.read_text())
    assert failed['checkpoint']==str(destination) and failed['load_count']==updated['load_count']
    # Replacing that same path with valid weights must retry and really reload.
    shutil.copyfile(first,destination)
    repaired=wait_status(lambda s:s.get('status')=='running' and s.get('load_count',0)>updated['load_count'])
    assert repaired['sha256']==digest(first)
    # Change the active stage. The same browser tab must reconnect by itself.
    publish_state('balance-050')
    changed=wait_status(lambda s:s.get('status')=='running' and s.get('stage')=='balance-050',seconds=90)
    wait_label('balance-050/model_99.pt',seconds=45)
    assert changed['goal_s']==.5 and page.url=='http://127.0.0.1:8088/'
    assert not errors,errors
    page.screenshot(path=str(ROOT/'local/live-viewer-validation.png'))
    report=dict(verified=True,new_checkpoint_auto_loaded=True,same_name_rewrite_auto_loaded=True,
        incomplete_checkpoint_preserved_previous_actor=True,stage_auto_followed=True,
        browser_refresh_required=False,stock_checkpoint_controls=True,javascript_errors=errors,
        initial=initial,updated=updated,repaired=repaired,stage_changed=changed)
    (ROOT/'local/live-viewer-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('initial','updated','repaired','stage_changed')},indent=2))
    browser.close()
