"""
Health Data Analysis - Digital Dashboard (Mini Project)
Run:  pip install numpy matplotlib
      python health_dashboard.py      (opens in your browser at http://127.0.0.1:8000)

- NumPy: cleaning, filtering, statistics, histograms, correlations
- Matplotlib: PDF report download
- Add / delete persons, CSV download. Data saved in health_data.csv
- No dataset needed: sample patients are generated on first run
"""
import csv, io, json, os, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

FILE = "health_data.csv"
COLS = ["id", "name", "age", "gender", "systolic_bp", "cholesterol", "blood_sugar", "heart_rate", "bmi"]
NUM = ["age", "systolic_bp", "cholesterol", "blood_sugar", "heart_rate", "bmi"]
RANGE = {"age": (1, 110), "systolic_bp": (70, 250), "cholesterol": (80, 500),
         "blood_sugar": (40, 500), "heart_rate": (30, 220), "bmi": (10, 60)}
lock = threading.Lock()


# ---------------- data ----------------
def sample(n=60, seed=42):
    g = np.random.default_rng(seed)
    fn = ["Aman", "Simran", "Rohit", "Priya", "Karan", "Neha", "Arjun", "Pooja", "Vikas", "Anjali",
          "Rahul", "Sneha", "Harpreet", "Manpreet", "Gurpreet", "Ritika", "Deepak", "Kavita"]
    ln = ["Sharma", "Singh", "Verma", "Gill", "Kaur", "Mehta", "Kumar", "Bansal"]
    age = np.clip(g.normal(45, 15, n), 18, 85).astype(int)
    bp = np.clip(95 + 0.6 * age + g.normal(0, 10, n), 90, 200).astype(int)
    ch = np.clip(140 + age + g.normal(0, 25, n), 110, 350).astype(int)
    su = np.clip(g.normal(105, 25, n) + 0.3 * age, 70, 300).astype(int)
    hr = np.clip(g.normal(75, 10, n), 50, 120).astype(int)
    bmi = np.round(np.clip(g.normal(25.5, 4, n), 16, 42), 1)
    sex = g.choice(["Male", "Female"], n)
    return [{"id": i + 1, "name": f"{g.choice(fn)} {g.choice(ln)}", "age": int(age[i]), "gender": str(sex[i]),
             "systolic_bp": int(bp[i]), "cholesterol": int(ch[i]), "blood_sugar": int(su[i]),
             "heart_rate": int(hr[i]), "bmi": float(bmi[i])} for i in range(n)]


def save(rs):
    with open(FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rs)


def load():
    if not os.path.exists(FILE):
        save(sample())
    with open(FILE, newline="") as f:
        return list(csv.DictReader(f))


rows = load()


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def arrays(rs):
    d = {k: np.array([_f(r.get(k)) for r in rs], dtype=float) for k in NUM}
    d["gender"] = np.array([r["gender"] for r in rs])
    return d


def clean(rs):
    """NumPy cleaning: drop rows with missing / impossible values."""
    d = arrays(rs)
    m = np.ones(len(rs), bool)
    for k, (lo, hi) in RANGE.items():
        m &= (d[k] >= lo) & (d[k] <= hi)  # NaN compares False -> removed
    return {k: v[m] for k, v in d.items()}


def select(q, f):
    """NumPy boolean-mask filtering + name search."""
    if not rows:
        return []
    d = arrays(rows)
    m = np.ones(len(rows), bool)
    if f == "highbp": m = d["systolic_bp"] >= 140
    elif f == "highchol": m = d["cholesterol"] >= 240
    elif f == "highsugar": m = d["blood_sugar"] >= 126
    elif f == "senior": m = d["age"] >= 60
    elif f in ("male", "female"): m = d["gender"] == f.capitalize()
    if q:
        m = m & np.array([q.lower() in r["name"].lower() for r in rows])
    return [r for r, k in zip(rows, m) if k]


def analytics():
    d = clean(rows)
    n = len(d["age"])
    if n == 0:
        return {"n": 0}
    bp = d["systolic_bp"]
    ah, _ = np.histogram(d["age"], bins=[0, 20, 30, 40, 50, 60, 70, 120])
    cat = [int((bp < 120).sum()), int(((bp >= 120) & (bp < 140)).sum()), int((bp >= 140).sum())]
    g, c = np.unique(d["gender"], return_counts=True)
    corr = np.nan_to_num(np.corrcoef(np.array([d[k] for k in NUM]))) if n > 2 else np.zeros((6, 6))
    return {"n": n, "avg": {k: round(float(d[k].mean()), 1) for k in NUM},
            "high_bp": round(100 * cat[2] / n, 1), "age_hist": ah.tolist(), "bp_cat": cat,
            "gender": dict(zip(g.tolist(), c.tolist())), "corr": corr.round(2).tolist(),
            "pts": {k: d[k].tolist() for k in NUM}}


def add(p):
    r = {"name": str(p.get("name", "")).strip(), "gender": p.get("gender")}
    if not r["name"]: raise ValueError("Name is required")
    if r["gender"] not in ("Male", "Female"): raise ValueError("Select gender")
    for k, (lo, hi) in RANGE.items():
        label = k.replace("_", " ")
        try: v = float(p.get(k))
        except (TypeError, ValueError): raise ValueError(f"Enter a valid {label}")
        if not lo <= v <= hi: raise ValueError(f"{label} must be between {lo} and {hi}")
        r[k] = round(v, 1) if k == "bmi" else int(v)
    with lock:
        r["id"] = max([int(x["id"]) for x in rows], default=0) + 1
        rows.append(r)
        save(rows)


def delete(i):
    with lock:
        n = len(rows)
        rows[:] = [r for r in rows if str(r["id"]) != str(i)]
        if len(rows) == n: raise ValueError("Person not found")
        save(rows)


def csv_text():
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
    return s.getvalue()


def make_pdf():
    d = clean(rows)
    if len(d["age"]) == 0: return None
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        f = Figure(figsize=(11, 8.5))
        a = f.subplots(2, 2)
        f.suptitle("Health Data Report - Demographics & Vitals", fontweight="bold")
        a[0, 0].hist(d["age"], bins=10, color="#00b8d4", edgecolor="white"); a[0, 0].set_title("Age Distribution")
        g, c = np.unique(d["gender"], return_counts=True)
        a[0, 1].bar(g, c, color=["#d500f9", "#00b8d4"][:len(g)]); a[0, 1].set_title("Gender Count")
        a[1, 0].hist(d["systolic_bp"], bins=10, color="#ff5252", edgecolor="white"); a[1, 0].set_title("Systolic BP")
        a[1, 1].hist(d["cholesterol"], bins=10, color="#ffab40", edgecolor="white"); a[1, 1].set_title("Cholesterol")
        f.tight_layout(rect=[0, 0, 1, .95]); pdf.savefig(f)
        f = Figure(figsize=(11, 8.5))
        a = f.subplots(2, 2)
        f.suptitle("Correlation Analysis", fontweight="bold")
        for ax, (x, y) in zip(a.ravel(), [("age", "systolic_bp"), ("age", "cholesterol"),
                                          ("systolic_bp", "cholesterol"), ("bmi", "blood_sugar")]):
            ax.scatter(d[x], d[y], color="#d500f9", alpha=.7, edgecolor="white")
            if len(d[x]) > 2:
                m, c0 = np.polyfit(d[x], d[y], 1)
                xs = np.linspace(d[x].min(), d[x].max(), 20)
                ax.plot(xs, m * xs + c0, color="black")
                ax.set_title(f"{x} vs {y} (r = {np.corrcoef(d[x], d[y])[0, 1]:+.2f})")
            ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=.3)
        f.tight_layout(rect=[0, 0, 1, .95]); pdf.savefig(f)
    return buf.getvalue()


# ---------------- server ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def out(self, body, ctype="application/json", code=200, hdr=None):
        if isinstance(body, (dict, list)): body = json.dumps(body)
        if isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (hdr or {}).items(): self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self.out(PAGE, "text/html; charset=utf-8")
        elif u.path == "/api/data":
            with lock:
                res = {"rows": select(q.get("q", [""])[0], q.get("f", ["all"])[0])[::-1],
                       "total": len(rows), "a": analytics()}
            self.out(res)
        elif u.path == "/api/csv":
            with lock: t = csv_text()
            self.out(t, "text/csv", hdr={"Content-Disposition": 'attachment; filename="health_data.csv"'})
        elif u.path == "/api/pdf":
            with lock: pdf = make_pdf()
            if pdf is None: return self.out({"error": "No valid data"}, code=400)
            self.out(pdf, "application/pdf", hdr={"Content-Disposition": 'attachment; filename="health_report.pdf"'})
        else:
            self.out({"error": "not found"}, code=404)

    def do_POST(self):
        try:
            p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/add": add(p)
            elif self.path == "/api/delete": delete(p.get("id"))
            else: return self.out({"error": "not found"}, code=404)
            self.out({"ok": True})
        except ValueError as e:
            self.out({"error": str(e)}, code=400)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Health Analytics</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#060a13;--card:rgba(13,21,38,.75);--line:rgba(0,229,255,.22);--cy:#00e5ff;--mg:#ff3df2;--gr:#39ff88;--rd:#ff4d6d;--tx:#cfe8ff;--mu:#6f88a8}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Rajdhani','Segoe UI',sans-serif;font-size:16px;color:var(--tx);min-height:100vh;
background:radial-gradient(900px 500px at 12% -10%,rgba(0,229,255,.15),transparent),radial-gradient(800px 500px at 100% 0,rgba(255,61,242,.13),transparent),var(--bg)}
body:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(0,229,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.05) 1px,transparent 1px);background-size:44px 44px}
.wrap{position:relative;max-width:1280px;margin:auto;padding:22px}
header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;margin-bottom:20px}
h1{font:900 22px Orbitron,monospace;letter-spacing:3px;color:var(--cy);text-shadow:0 0 18px rgba(0,229,255,.7)}
h1 span{color:var(--mg);text-shadow:0 0 18px rgba(255,61,242,.7)}
.st{display:flex;align-items:center;gap:10px;font:500 12px Orbitron,monospace;color:var(--mu);letter-spacing:1px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--gr);box-shadow:0 0 10px var(--gr);animation:p 1.6s infinite}
@keyframes p{50%{opacity:.3}}
.grid{display:grid;gap:16px}.k5{grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin-bottom:16px}
.c3{grid-template-columns:2fr 1fr 1fr;margin-bottom:16px}.c2{grid-template-columns:2fr 1fr;margin-bottom:16px}
@media(max-width:900px){.c3,.c2{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;backdrop-filter:blur(8px);box-shadow:0 0 24px rgba(0,229,255,.06),inset 0 0 30px rgba(0,229,255,.03)}
.card h3{font:700 11px Orbitron,monospace;letter-spacing:2px;color:var(--mu);margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.kpi b{display:block;font:900 30px Orbitron,monospace;color:var(--cy);text-shadow:0 0 16px rgba(0,229,255,.6);margin-top:6px}
.kpi:nth-child(2) b{color:var(--mg);text-shadow:0 0 16px rgba(255,61,242,.6)}.kpi:nth-child(4) b{color:var(--gr);text-shadow:0 0 16px rgba(57,255,136,.5)}.kpi:nth-child(5) b{color:var(--rd);text-shadow:0 0 16px rgba(255,77,109,.5)}
.ch{height:250px;position:relative}.ch.big{height:300px}
.btn{font:600 14px Rajdhani,sans-serif;letter-spacing:1px;text-transform:uppercase;text-decoration:none;color:var(--cy);background:transparent;border:1px solid var(--cy);border-radius:8px;padding:8px 16px;cursor:pointer;transition:.2s;display:inline-block}
.btn:hover{background:rgba(0,229,255,.12);box-shadow:0 0 16px rgba(0,229,255,.5)}
.btn.pri{background:linear-gradient(90deg,var(--cy),#3b82f6);color:#02101a;border:0;font-weight:700}
.btn.pri:hover{box-shadow:0 0 22px rgba(0,229,255,.7)}
.btn.mg{color:var(--mg);border-color:var(--mg)}.btn.mg:hover{background:rgba(255,61,242,.12);box-shadow:0 0 16px rgba(255,61,242,.5)}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input,select{background:rgba(0,0,0,.35);border:1px solid var(--line);color:var(--tx);border-radius:8px;padding:8px 12px;font:500 15px Rajdhani,sans-serif;outline:0}
input:focus,select:focus{border-color:var(--cy);box-shadow:0 0 10px rgba(0,229,255,.4)}
select option{background:#0b1426}
.tw{max-height:380px;overflow:auto;margin-top:12px;border-radius:10px}
table{width:100%;border-collapse:collapse}
th{position:sticky;top:0;background:#0b1426;font:700 11px Orbitron,monospace;letter-spacing:1px;color:var(--cy);text-align:left;padding:11px 10px}
td{padding:9px 10px;border-top:1px solid rgba(0,229,255,.08)}tr:hover td{background:rgba(0,229,255,.06)}
td.hi{color:var(--rd);font-weight:700}td.mu{color:var(--mu);text-align:center;padding:30px}
.x{background:0;border:1px solid rgba(255,77,109,.5);color:var(--rd);border-radius:6px;width:28px;height:28px;cursor:pointer}.x:hover{background:rgba(255,77,109,.2)}
#hm{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;font-size:12px;text-align:center}#hm b{color:var(--mu);font:700 9px Orbitron,monospace;align-self:center}#hm span{padding:9px 0;border-radius:5px}
#rv{font:700 12px Orbitron,monospace;color:var(--mg)}
#md{position:fixed;inset:0;background:rgba(2,6,14,.8);backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center;z-index:9}#md.on{display:flex}
#md form{width:min(480px,92vw)}.fg{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.fg label{font-size:12px;color:var(--mu);letter-spacing:1px;text-transform:uppercase;display:grid;gap:5px}.fg input,.fg select{width:100%}
#toast{position:fixed;bottom:24px;right:24px;padding:12px 20px;border-radius:10px;background:#0b1426;border:1px solid var(--gr);color:var(--gr);transform:translateY(90px);opacity:0;transition:.3s;z-index:20}
#toast.show{transform:none;opacity:1}#toast.bad{border-color:var(--rd);color:var(--rd)}
</style></head><body><div class="wrap">
<header><h1>HEALTH<span>//</span>ANALYTICS</h1>
<div class="st"><span class="dot"></span>SYSTEM ONLINE <span id="clk"></span></div></header>

<div class="grid k5">
<div class="card kpi"><h3>TOTAL PATIENTS</h3><b id="k_total">0</b></div>
<div class="card kpi"><h3>AVG AGE</h3><b id="k_age">-</b></div>
<div class="card kpi"><h3>AVG BP (mmHg)</h3><b id="k_bp">-</b></div>
<div class="card kpi"><h3>AVG CHOLESTEROL</h3><b id="k_chol">-</b></div>
<div class="card kpi"><h3>HIGH BP RATE</h3><b id="k_hbp">-</b></div></div>

<div class="grid c3">
<div class="card"><h3>AGE DISTRIBUTION</h3><div class="ch"><canvas id="c_age"></canvas></div></div>
<div class="card"><h3>GENDER</h3><div class="ch"><canvas id="c_gen"></canvas></div></div>
<div class="card"><h3>BLOOD PRESSURE LEVEL</h3><div class="ch"><canvas id="c_bp"></canvas></div></div></div>

<div class="grid c2">
<div class="card"><h3><span>CORRELATION SCATTER</span><span class="bar"><select id="sx"></select><select id="sy"></select><span id="rv"></span></span></h3><div class="ch big"><canvas id="c_sc"></canvas></div></div>
<div class="card"><h3>CORRELATION MATRIX</h3><div id="hm"></div></div></div>

<div class="card"><h3><span>PATIENT RECORDS <span id="cnt" style="color:var(--cy)"></span></span>
<span class="bar"><a class="btn" href="/api/csv" download>⬇ CSV</a><a class="btn mg" href="/api/pdf" download>⬇ PDF Report</a><button class="btn pri" onclick="md.classList.add('on')">+ Add Person</button></span></h3>
<div class="bar"><input id="q" placeholder="Search name..." oninput="load(0)"><select id="f" onchange="load(0)">
<option value="all">All patients</option><option value="highbp">High BP (≥140)</option><option value="highchol">High cholesterol (≥240)</option><option value="highsugar">High sugar (≥126)</option><option value="senior">Age 60+</option><option value="male">Male</option><option value="female">Female</option></select></div>
<div class="tw"><table><thead><tr><th>ID</th><th>NAME</th><th>AGE</th><th>SEX</th><th>BP</th><th>CHOL</th><th>SUGAR</th><th>HR</th><th>BMI</th><th></th></tr></thead><tbody id="tb"></tbody></table></div></div></div>

<div id="md" onclick="if(event.target===md)md.classList.remove('on')"><form class="card" onsubmit="addP(event)">
<h3 style="font-size:14px;color:var(--cy)">ADD NEW PERSON</h3><div class="fg" id="fg"></div>
<div class="bar" style="justify-content:flex-end"><button type="button" class="btn" onclick="md.classList.remove('on')">Cancel</button><button class="btn pri">Save</button></div></form></div>
<div id="toast"></div>

<script>
const $=s=>document.querySelector(s),md=$("#md"),L={age:"Age",systolic_bp:"Systolic BP",cholesterol:"Cholesterol",blood_sugar:"Blood Sugar",heart_rate:"Heart Rate",bmi:"BMI"},K=Object.keys(L),
R={age:[1,110],systolic_bp:[70,250],cholesterol:[80,500],blood_sugar:[40,500],heart_rate:[30,220],bmi:[10,60]};
let D,C={};
Chart.defaults.color="#8fb0d0";Chart.defaults.font.family="Rajdhani";Chart.defaults.borderColor="rgba(0,229,255,.1)";
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function mk(id,cfg){if(C[id])C[id].destroy();C[id]=new Chart($("#"+id),cfg)}
function toast(m,bad){const t=$("#toast");t.textContent=m;t.className="show"+(bad?" bad":"");setTimeout(()=>t.className="",2400)}
async function load(full=1){D=await(await fetch(`/api/data?q=${encodeURIComponent($("#q").value)}&f=${$("#f").value}`)).json();table();if(full){kpis();charts();scat();heat()}}
function kpis(){const a=D.a;$("#k_total").textContent=D.total;if(!a.n)return;$("#k_age").textContent=a.avg.age;$("#k_bp").textContent=a.avg.systolic_bp;$("#k_chol").textContent=a.avg.cholesterol;$("#k_hbp").textContent=a.high_bp+"%"}
function table(){$("#tb").innerHTML=D.rows.map(r=>`<tr><td>#${r.id}</td><td>${esc(r.name)}</td><td>${r.age}</td><td>${r.gender}</td><td class="${r.systolic_bp>=140?"hi":""}">${r.systolic_bp}</td><td class="${r.cholesterol>=240?"hi":""}">${r.cholesterol}</td><td class="${r.blood_sugar>=126?"hi":""}">${r.blood_sugar}</td><td>${r.heart_rate}</td><td>${r.bmi}</td><td><button class="x" onclick="del(${r.id})">✕</button></td></tr>`).join("")||'<tr><td colspan="10" class="mu">No records found</td></tr>';$("#cnt").textContent=`[${D.rows.length}/${D.total}]`}
function charts(){const a=D.a;if(!a.n)return;
mk("c_age",{type:"bar",data:{labels:["<20","20s","30s","40s","50s","60s","70+"],datasets:[{data:a.age_hist,backgroundColor:"rgba(0,229,255,.5)",borderColor:"#00e5ff",borderWidth:1.5,borderRadius:6}]},options:{maintainAspectRatio:false,plugins:{legend:{display:false}}}});
mk("c_gen",{type:"doughnut",data:{labels:Object.keys(a.gender),datasets:[{data:Object.values(a.gender),backgroundColor:["#ff3df2","#00e5ff"],borderColor:"#060a13",borderWidth:3}]},options:{maintainAspectRatio:false,cutout:"68%"}});
mk("c_bp",{type:"doughnut",data:{labels:["Normal","Elevated","High"],datasets:[{data:a.bp_cat,backgroundColor:["#39ff88","#ffb020","#ff4d6d"],borderColor:"#060a13",borderWidth:3}]},options:{maintainAspectRatio:false,cutout:"68%"}})}
function scat(){const a=D.a;if(!a.n)return;const x=$("#sx").value,y=$("#sy").value,X=a.pts[x],Y=a.pts[y],n=X.length,mx=X.reduce((s,v)=>s+v,0)/n,my=Y.reduce((s,v)=>s+v,0)/n;
let sxy=0,sxx=0;for(let i=0;i<n;i++){sxy+=(X[i]-mx)*(Y[i]-my);sxx+=(X[i]-mx)**2}
const m=sxx?sxy/sxx:0,c=my-m*mx,lo=Math.min(...X),hi=Math.max(...X),r=a.corr[K.indexOf(x)][K.indexOf(y)];
$("#rv").textContent="r = "+(r>0?"+":"")+r.toFixed(2);
mk("c_sc",{type:"scatter",data:{datasets:[{data:X.map((v,i)=>({x:v,y:Y[i]})),backgroundColor:"rgba(255,61,242,.75)",pointRadius:5},{type:"line",data:[{x:lo,y:m*lo+c},{x:hi,y:m*hi+c}],borderColor:"#00e5ff",borderWidth:2,pointRadius:0}]},options:{maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:L[x]}},y:{title:{display:true,text:L[y]}}}}})}
function heat(){const a=D.a;if(!a.n)return;const s=k=>L[k].split(" ").pop();let h="<div></div>"+K.map(k=>`<b>${s(k)}</b>`).join("");
K.forEach((k,i)=>{h+=`<b>${s(k)}</b>`+a.corr[i].map(v=>`<span style="background:${v>=0?"rgba(0,229,255,":"rgba(255,61,242,"}${(Math.abs(v)*.75).toFixed(2)})">${v.toFixed(2)}</span>`).join("")});$("#hm").innerHTML=h}
async function addP(e){e.preventDefault();const p=Object.fromEntries(new FormData(e.target)),r=await fetch("/api/add",{method:"POST",body:JSON.stringify(p)}),j=await r.json();
if(!r.ok)return toast(j.error,1);e.target.reset();md.classList.remove("on");toast("Person added");load()}
async function del(id){if(!confirm("Delete patient #"+id+"?"))return;const r=await fetch("/api/delete",{method:"POST",body:JSON.stringify({id})});toast(r.ok?"Person deleted":"Delete failed",!r.ok);load()}
$("#sx").innerHTML=$("#sy").innerHTML=K.map(k=>`<option value="${k}">${L[k]}</option>`).join("");$("#sx").value="age";$("#sy").value="systolic_bp";$("#sx").onchange=$("#sy").onchange=scat;
$("#fg").innerHTML='<label style="grid-column:span 2">Name<input name="name" required></label><label>Gender<select name="gender"><option>Male</option><option>Female</option></select></label>'+K.map(k=>`<label>${L[k]} (${R[k][0]}-${R[k][1]})<input name="${k}" type="number" step="${k=="bmi"?"0.1":"1"}" min="${R[k][0]}" max="${R[k][1]}" required></label>`).join("");
setInterval(()=>$("#clk").textContent=new Date().toLocaleTimeString(),1000);load();
</script></body></html>"""


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8000), H)
    print("Dashboard running at http://127.0.0.1:8000  (Ctrl+C to stop)")
    threading.Timer(1, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
