# app.py — Painel Habisolute (compacto e funcional)
# Execução: streamlit run app.py
# TV: ?tv=1&interval=25

import os, io, re
from datetime import date, datetime
from typing import Optional, Dict, Any, List

# ---------------- Streamlit ----------------
HAS_ST = True
try:
    import streamlit as st
    import streamlit.components.v1 as components
except Exception:
    HAS_ST = False

def _rerun():
    if not HAS_ST:
        return
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---------------- DB / ORM ----------------
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB = os.getenv("HABI_DB", "habisolute_painel.db")
engine = create_engine(f"sqlite:///{DB}", future=True)
Session = sessionmaker(bind=engine)
Base = declarative_base()

STATUS_AG_CONFERENCIA = "Aguardando conferência"
STATUS_AG_RUPTURA      = "Aguardando ruptura"
STATUS_ROMPIDO         = "Rompido"

class Specimen(Base):
    __tablename__ = "specimens"
    id = Column(Integer, primary_key=True)
    tipo = Column(String(20), default="CP", nullable=False)
    obra = Column(String(200))
    cp_codigo = Column(String(80))
    idade_alvo_dias = Column(Integer, default=28)
    data_moldagem = Column(Date, default=date.today, nullable=False)
    data_prevista = Column(Date, default=date.today, nullable=False)
    status = Column(String(30), default=STATUS_AG_CONFERENCIA, nullable=False)
    responsavel = Column(String(120))
    observacoes = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Meta(Base):
    __tablename__ = "metas"
    id = Column(Integer, primary_key=True)
    data = Column(Date, nullable=False)
    obra = Column(String(200))
    meta_texto = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

with engine.begin() as c:
    c.exec_driver_sql("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")

def set_setting(key: str, value: str):
    with engine.begin() as c:
        c.exec_driver_sql(
            "INSERT INTO settings(key,value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

def get_setting(key: str) -> Optional[str]:
    try:
        with engine.begin() as c:
            r = list(c.exec_driver_sql("SELECT value FROM settings WHERE key=?", (key,)))
            return r[0][0] if r else None
    except Exception:
        return None

def set_current_operator(nome: Optional[str]):
    if nome:
        set_setting("op", nome)

def get_current_operator() -> Optional[str]:
    return get_setting("op")

# ---------------- Helpers ----------------
CP_RE = re.compile(r"\b\d{3}\.\d{3}\b")

def normalize_cp(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    m = CP_RE.search(s)
    if m:
        return m.group(0)
    digits = re.sub(r"\D", "", s)
    if len(digits) == 6:
        return f"{digits[:3]}.{digits[3:]}"
    return s

def load_df(tipos: Optional[List[str]] = None) -> pd.DataFrame:
    with Session() as s:
        df = pd.read_sql_table("specimens", s.bind)

    if df.empty:
        return df

    for c in ("data_prevista", "data_moldagem", "created_at", "updated_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if tipos:
        df = df[df["tipo"].isin(tipos)]

    h_ts = pd.Timestamp(date.today()).normalize()
    dp = pd.to_datetime(df["data_prevista"], errors="coerce").dt.normalize()
    diff = (dp - h_ts).dt.days
    df["SLA"] = diff.apply(
        lambda v: "Atrasado" if pd.notna(v) and v < 0
        else ("Hoje" if v == 0 else ("Próx 7d" if pd.notna(v) and v <= 7 else ">7d"))
    )
    return df

def upsert_codigo(data: Dict[str, Any]):
    code = normalize_cp(data.get("cp_codigo") or "")
    if not code:
        return
    data["cp_codigo"] = code
    with Session() as s:
        sp = s.query(Specimen).filter(Specimen.cp_codigo == code).one_or_none()
        if sp:
            for k, v in data.items():
                setattr(sp, k, v)
            sp.updated_at = datetime.utcnow()
        else:
            s.add(Specimen(**data))
        s.commit()

def marcar_status(code: str, status: str, operador: Optional[str] = None) -> int:
    code = normalize_cp(code)
    if not code:
        return 0
    with Session() as s:
        q = s.query(Specimen).filter(Specimen.cp_codigo == code)
        n = 0
        for sp in q:
            # só conta se realmente houver mudança (evita “sucesso” falso)
            if sp.status != status or (operador and sp.responsavel != operador):
                sp.status = status
                if operador:
                    sp.responsavel = operador
                sp.updated_at = datetime.utcnow()
                n += 1
        s.commit()
        return n

def fetch_by_code_exact(code: str) -> pd.DataFrame:
    code = normalize_cp(code)
    if not code:
        return pd.DataFrame()
    with Session() as s:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM specimens WHERE cp_codigo=?",
                s.bind,
                params=(code,),
            )
        except Exception:
            return pd.DataFrame()
    for c in ("data_prevista", "data_moldagem", "created_at", "updated_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def search_codes(term: str, limit: int = 20) -> pd.DataFrame:
    like = f"%{term}%"
    with Session() as s:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM specimens WHERE cp_codigo LIKE ? ORDER BY updated_at DESC, id DESC LIMIT ?",
                s.bind,
                params=(like, limit),
            )
        except Exception:
            return pd.DataFrame()
    for c in ("data_prevista", "data_moldagem", "created_at", "updated_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def fetch_last5(limit: int = 5) -> pd.DataFrame:
    with Session() as s:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM specimens ORDER BY updated_at DESC, id DESC LIMIT ?",
                s.bind,
                params=(limit,),
            )
        except Exception:
            return pd.DataFrame()
    for c in ("data_prevista", "data_moldagem", "created_at", "updated_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# metas
def add_meta(data_meta: date, obra: Optional[str], texto: str):
    tx = (texto or "").strip()
    if not tx:
        return 0
    with Session() as s:
        s.add(
            Meta(
                data=pd.to_datetime(data_meta).date(),
                obra=(obra or None),
                meta_texto=tx,
                created_at=datetime.utcnow(),
            )
        )
        s.commit()
        return 1

def metas_do_dia(d: date) -> pd.DataFrame:
    with Session() as s:
        q = pd.read_sql_query(
            "SELECT data, obra, meta_texto AS meta, created_at AS criado "
            "FROM metas WHERE date(data)=date(?) ORDER BY created_at ASC",
            s.bind,
            params=(pd.to_datetime(d).date(),),
        )
    if not q.empty:
        q["criado"] = pd.to_datetime(q["criado"], errors="coerce")
    return q

def metas_ultimas(limit: int = 10) -> pd.DataFrame:
    with Session() as s:
        q = pd.read_sql_query(
            "SELECT data, obra, meta_texto AS meta, created_at AS criado "
            "FROM metas ORDER BY created_at DESC LIMIT ?",
            s.bind,
            params=(limit,),
        )
    if not q.empty:
        q["criado"] = pd.to_datetime(q["criado"], errors="coerce")
    return q

def wipe_all() -> int:
    total = 0
    with engine.begin() as c:
        total += c.exec_driver_sql("DELETE FROM specimens").rowcount or 0
        total += c.exec_driver_sql("DELETE FROM metas").rowcount or 0
        total += c.exec_driver_sql("DELETE FROM settings").rowcount or 0
    # VACUUM fora de transação (SQLite reclama dentro de begin)
    try:
        with engine.connect() as c:
            c.exec_driver_sql("VACUUM")
    except Exception:
        pass
    return total

# -------- PDF (import simples) --------
try:
    import pdfplumber
except Exception:
    pdfplumber = None

def import_pdf_codes(file_bytes: bytes, status: str = STATUS_AG_CONFERENCIA) -> int:
    if not pdfplumber:
        return 0
    count = 0
    hoje = pd.Timestamp(date.today()).normalize()
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            for code in set(CP_RE.findall(txt)):
                upsert_codigo(
                    {
                        "tipo": "CP",
                        "cp_codigo": code,
                        "data_moldagem": (hoje - pd.Timedelta(days=28)).date(),
                        "idade_alvo_dias": 28,
                        "data_prevista": hoje.date(),
                        "status": status,
                    }
                )
                count += 1
    return count

# --------------- UI -----------------
if HAS_ST:
    st.set_page_config(page_title="Painel — Habisolute", layout="wide", page_icon="🧪")

    # ---------------- Query params (compatível) ----------------
    def _qp_get(qp: Any, key: str, default: str) -> str:
        """
        Compatível com:
        - st.experimental_get_query_params(): dict[str, list[str]]
        - st.query_params: pode retornar str, list, ou objeto dict-like
        """
        try:
            v = qp.get(key, None)
        except Exception:
            v = None

        if v is None:
            return default

        # caso seja lista/tupla (experimental_get_query_params)
        if isinstance(v, (list, tuple)):
            return str(v[0]) if v else default

        # caso st.query_params retorne string
        return str(v)

    try:
        qp_obj = st.query_params
        qp = dict(qp_obj)  # dict-like
    except Exception:
        qp = st.experimental_get_query_params()

    TV = _qp_get(qp, "tv", "0").lower() in ("1", "true")
    try:
        INTERVAL = int(_qp_get(qp, "interval", "25"))
    except Exception:
        INTERVAL = 25
    try:
        IDX = int(_qp_get(qp, "tv_page", "0"))
    except Exception:
        IDX = 0

    # tema
    if "theme" not in st.session_state:
        st.session_state["theme"] = get_setting("theme") or "Escuro"

    with st.sidebar:
        st.markdown("### Aparência")
        theme_choice = st.radio(
            "Tema",
            ["Escuro", "Claro"],
            index=0 if st.session_state["theme"] == "Escuro" else 1,
            horizontal=True,
        )
        if theme_choice != st.session_state["theme"]:
            st.session_state["theme"] = theme_choice
            set_setting("theme", theme_choice)
            _rerun()

    THEME_DARK = """
    <style>
      :root { --accent:#f97316; --stroke:rgba(255,255,255,.14); --txt:#e5e7eb; --bg1:#0b1220; --bg2:#060b14; --bg3:#03060b; }
      .stApp { background: radial-gradient(1000px 600px at 10% 10%, var(--bg1) 0%, var(--bg2) 60%, var(--bg3) 100%); color:var(--txt); }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(17,24,39,.96), rgba(12,18,30,.96)); border-right:1px solid var(--stroke); }
      [data-testid="stSidebar"] * { color:var(--txt) !important; }
      .stButton>button{ border:1px solid var(--stroke); border-radius:12px; box-shadow:0 8px 18px rgba(249,115,22,.18); }
      .stDataFrame { border:1px solid var(--stroke); border-radius:14px; }
      h1,h2,h3{ color:#fff }
    </style>"""
    THEME_LIGHT = """
    <style>
      :root { --accent:#f97316; --stroke:rgba(0,0,0,.10); --txt:#0f172a; --bg:#f9fafb; --card:#ffffff; }
      .stApp { background: linear-gradient(180deg, var(--bg), #eef2f7); color:var(--txt); }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff, #f6f7fb); border-right:1px solid var(--stroke); }
      .stButton>button{ border:1px solid var(--stroke); border-radius:12px; box-shadow:0 8px 18px rgba(249,115,22,.12); }
      .stDataFrame { border:1px solid var(--stroke); border-radius:14px; background:var(--card); }
      h1,h2,h3{ color:#0f172a }
    </style>"""
    st.markdown(
        THEME_DARK if st.session_state["theme"] == "Escuro" else THEME_LIGHT,
        unsafe_allow_html=True,
    )

    # filtros
    if not TV:
        st.sidebar.header("Filtros")
        tipos = st.sidebar.multiselect(
            "Tipos", ["CP", "Prisma", "Argamassa"], default=["CP", "Prisma", "Argamassa"]
        )
        f_obra = st.sidebar.text_input("Obra contém")
    else:
        tipos = ["CP", "Prisma", "Argamassa"]
        f_obra = ""

    df = load_df(tipos)
    if not df.empty and f_obra:
        df = df[df["obra"].fillna("").str.contains(f_obra, case=False, na=False)]

    hoje_ts = pd.Timestamp(date.today()).normalize()
    dp_all = (
        pd.to_datetime(df["data_prevista"], errors="coerce").dt.normalize()
        if not df.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    df_today = df[dp_all == hoje_ts] if not df.empty else df
    df_over = df[dp_all < hoje_ts] if not df.empty else df
    df_next7 = (
        df[(dp_all > hoje_ts) & (dp_all <= hoje_ts + pd.Timedelta(days=7))]
        if not df.empty
        else df
    )

    def grid(d: pd.DataFrame, title: str, h: int = 420):
        st.subheader(title)
        if d.empty:
            st.info("Sem itens.")
            return
        cols = [
            c
            for c in [
                "tipo",
                "obra",
                "cp_codigo",
                "idade_alvo_dias",
                "data_moldagem",
                "data_prevista",
                "status",
                "SLA",
                "observacoes",
            ]
            if c in d.columns
        ]
        sort_cols = [c for c in ["data_prevista", "obra", "cp_codigo"] if c in cols]
        st.dataframe(
            d[cols].sort_values(sort_cols) if sort_cols else d[cols],
            use_container_width=True,
            height=h,
        )

    if not TV:
        a, b, c, dcol = st.columns(4)
        with a:
            st.metric("Hoje", len(df_today))
        with b:
            st.metric("Atrasados", len(df_over))
        with c:
            st.metric("Próx. 7 dias", len(df_next7))
        with dcol:
            total_ativo = 0
            if not df.empty and "status" in df.columns:
                total_ativo = int((df["status"].fillna("") != STATUS_ROMPIDO).sum())
            st.metric("Total ativo", total_ativo)

        t1, t2, t3, t4, timp, t_cole, toper, t_metas, t_tools = st.tabs(
            [
                "Hoje",
                "Atrasados",
                "Próx. 7 dias",
                "Todos",
                "Importar",
                "Coletas",
                "Operador",
                "Metas",
                "Ferramentas",
            ]
        )
        with t1:
            grid(df_today, "Hoje", 360)
        with t2:
            grid(df_over, "Atrasados", 520)
        with t3:
            grid(df_next7, "Próx. 7 dias", 520)
        with t4:
            grid(df, "Todos", 540)

        # ---------- Importar ----------
        with timp:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("PDF de Coleta (marca 'Aguardando conferência')")
                up = st.file_uploader("Coleta", type=["pdf"], key="coleta")
                if up:
                    st.success(
                        f"{import_pdf_codes(up.read(), STATUS_AG_CONFERENCIA)} registro(s) processado(s)."
                    )
                    _rerun()
            with c2:
                st.caption("PDF de Ruptura (marca 'Rompido')")
                up2 = st.file_uploader("Ruptura", type=["pdf"], key="ruptura")
                if up2:
                    st.success(
                        f"{import_pdf_codes(up2.read(), STATUS_ROMPIDO)} registro(s) processado(s)."
                    )
                    _rerun()

        # ---------- Coletas ----------
        with t_cole:
            st.subheader("Coletas — Conferência de chegada")
            st.caption(
                "Fluxo: Importa lista → conferência ao chegar → marcar **Aguardando ruptura**."
            )
            op_atual = get_current_operator()

            if "cp_buf" not in st.session_state:
                st.session_state["cp_buf"] = ""

            cpad1, cpad2, _ = st.columns([1, 1, 2])
            cpad1.text_input("CP", key="cp_buf", placeholder="000.000")

            def _append(lbl: str):
                buf = st.session_state.get("cp_buf", "")
                st.session_state["cp_buf"] = buf[:-1] if lbl == "⌫" else buf + lbl
                _rerun()

            with cpad2:
                for row in [["7", "8", "9"], ["4", "5", "6"], ["1", "2", "3"], [".", "0", "⌫"]]:
                    cA, cB, cC = st.columns(3)
                    if cA.button(row[0], use_container_width=True):
                        _append(row[0])
                    if cB.button(row[1], use_container_width=True):
                        _append(row[1])
                    if cC.button(row[2], use_container_width=True):
                        _append(row[2])

            colA, colB = st.columns([1, 1])
            if colA.button("Limpar"):
                st.session_state["cp_buf"] = ""
                st.session_state["last_auto_search"] = ""
                _rerun()

            # auto-busca: só dispara quando o buffer muda (evita loop infinito)
            pressed = colB.button("🔎 Buscar")
            raw_buf = (st.session_state.get("cp_buf") or "").strip()
            last_auto = st.session_state.get("last_auto_search", "")
            auto_fire = bool(raw_buf) and len(raw_buf) >= 3 and raw_buf != last_auto
            do_search = pressed or auto_fire
            if auto_fire:
                st.session_state["last_auto_search"] = raw_buf

            if not df.empty:
                dp_all_local = pd.to_datetime(df["data_prevista"], errors="coerce").dt.normalize()
                aguardando = df[(df["status"] == STATUS_AG_CONFERENCIA) & (dp_all_local == hoje_ts)]
            else:
                aguardando = df

            options_aguard = aguardando["cp_codigo"].astype(str).tolist() if not aguardando.empty else []

            # estado interno
            if "sel_cole_vals" not in st.session_state:
                st.session_state["sel_cole_vals"] = []

            # filtra seleção interna para evitar erro “default fora das options”
            st.session_state["sel_cole_vals"] = [
                x for x in st.session_state["sel_cole_vals"] if x in options_aguard
            ]

            # se pedimos limpeza no ciclo anterior, remova a key do widget antes de instanciar
            if st.session_state.get("clear_sel_cole", False):
                st.session_state.pop("sel_cole_widget", None)
                st.session_state["clear_sel_cole"] = False

            # BUSCA → adiciona à seleção interna E ao widget (corrige bug)
            if do_search:
                target = normalize_cp(raw_buf)
                det = fetch_by_code_exact(target)
                if det.empty:
                    if pressed:
                        st.warning("CP não encontrado no banco.")
                else:
                    cols_show = [
                        c
                        for c in [
                            "cp_codigo",
                            "obra",
                            "data_moldagem",
                            "data_prevista",
                            "idade_alvo_dias",
                            "responsavel",
                            "status",
                        ]
                        if c in det.columns
                    ]
                    st.dataframe(det[cols_show], use_container_width=True, height=120)

                if target in options_aguard:
                    cur = st.session_state.get("sel_cole_widget", st.session_state["sel_cole_vals"]) or []
                    cur = [x for x in cur if x in options_aguard]
                    if target not in cur:
                        cur = cur + [target]
                        st.session_state["sel_cole_vals"] = cur
                        # garante que o widget reflita na hora
                        st.session_state["sel_cole_widget"] = cur
                        st.success(f"CP {target} adicionado à seleção para '{STATUS_AG_RUPTURA}'.")
                        _rerun()
                else:
                    if not det.empty and target and pressed:
                        st.info("Este CP não está na lista 'Aguardando conferência (para hoje)'.")

            with st.expander(f"Aguardando conferência (para hoje) — {len(aguardando)}"):
                if aguardando.empty:
                    st.caption("Nenhum CP para hoje.")
                else:
                    st.dataframe(
                        aguardando[["cp_codigo", "obra", "data_moldagem", "data_prevista", "status"]],
                        use_container_width=True,
                        height=220,
                    )

            # Multiselect: key estável, default = estado interno (já filtrado)
            sel = st.multiselect(
                "Selecionar CPs para marcar como Aguardando ruptura",
                options_aguard,
                default=st.session_state["sel_cole_vals"],
                key="sel_cole_widget",
            )
            # sincroniza interno com o widget (evita divergência)
            st.session_state["sel_cole_vals"] = [x for x in (sel or []) if x in options_aguard]

            c1, c2 = st.columns([1, 1])
            if c1.button("✅ Marcar Aguardando ruptura (selecionados)", type="primary"):
                selected = st.session_state.get("sel_cole_widget", []) or []
                total = 0
                for ccode in selected:
                    total += marcar_status(str(ccode), STATUS_AG_RUPTURA, op_atual)
                if total > 0:
                    st.success(f"{total} registro(s) atualizado(s) para '{STATUS_AG_RUPTURA}'.")
                    # limpa interno e agenda limpeza do widget para o próximo ciclo
                    st.session_state["sel_cole_vals"] = []
                    st.session_state["clear_sel_cole"] = True
                    _rerun()
                else:
                    st.warning("Nenhum registro foi atualizado. Selecione um ou mais CPs.")

            code_quick = c2.text_input("Marcar rápido (digite um CP)", key="cole_quick")
            if c2.button("✔️ Marcar 1 CP") and code_quick:
                n = marcar_status(code_quick.strip(), STATUS_AG_RUPTURA, op_atual)
                if n > 0:
                    st.success(f"{n} registro(s) atualizado(s) para '{STATUS_AG_RUPTURA}'.")
                    _rerun()
                else:
                    st.warning("CP não encontrado ou já está com o status solicitado.")

        # ---------- Operador ----------
        with toper:
            st.subheader("Operador & Dar baixa")
            op_atual = get_current_operator()
            colA, colB, colC = st.columns([1, 1, 1])
            novo = colA.text_input("Operador atual", value=op_atual or "")
            if colB.button("📌 Salvar operador"):
                set_current_operator(novo)
                st.success(f"Operador: {novo}")
            code = colC.text_input("Buscar CP (ex.: 037.421)")
            if code:
                results = search_codes(code.strip(), 10)
                if results.empty:
                    st.warning("Nenhum CP encontrado.")
                else:
                    st.markdown("#### Resultado da busca")
                    opts = [
                        f"{r.cp_codigo} — {r.obra or 's/obra'} — {r.status}"
                        for r in results.itertuples(index=False)
                    ]
                    sel = st.selectbox("Selecione um CP", opts, index=0, key="op_sel_code")
                    chosen = results.iloc[opts.index(sel)]["cp_codigo"]
                    det2 = fetch_by_code_exact(chosen)
                    cols = [
                        c
                        for c in [
                            "tipo",
                            "obra",
                            "cp_codigo",
                            "idade_alvo_dias",
                            "data_moldagem",
                            "data_prevista",
                            "status",
                            "responsavel",
                            "observacoes",
                        ]
                        if c in det2.columns
                    ]
                    st.dataframe(det2[cols], use_container_width=True, height=140)
                    b1, b2, b3 = st.columns(3)
                    if b1.button("✅ Rompido"):
                        st.success(f"{marcar_status(chosen, STATUS_ROMPIDO, novo)} atualizado(s).")
                        _rerun()
                    if b2.button("⏳ Aguardando ruptura"):
                        st.success(f"{marcar_status(chosen, STATUS_AG_RUPTURA, novo)} atualizado(s).")
                        _rerun()
                    if b3.button("↩️ Aguardando conferência"):
                        st.success(f"{marcar_status(chosen, STATUS_AG_CONFERENCIA, novo)} atualizado(s).")
                        _rerun()

            st.divider()
            st.markdown("### Últimos 5 CPs (recentes)")
            last5 = fetch_last5(5)
            if last5.empty:
                st.caption("Sem registros recentes.")
            else:
                cols5 = [c for c in ["cp_codigo", "obra", "data_prevista", "status", "responsavel", "updated_at"] if c in last5.columns]
                st.dataframe(last5[cols5], use_container_width=True, height=200)

        # ---------- Metas ----------
        with t_metas:
            st.subheader("Metas do dia")
            with st.form("form_meta"):
                dt_meta = st.date_input("Data", value=date.today(), format="DD/MM/YYYY")
                obra_meta = st.text_input("Obra (opcional)")
                meta_texto = st.text_area("Texto da meta", placeholder="Ex.: Romper CPs da Obra X às 10h")
                ok = st.form_submit_button("Salvar meta", type="primary")
                if ok:
                    n = add_meta(dt_meta, obra_meta, meta_texto)
                    st.success("Meta salva.") if n else st.warning("Informe o texto.")
                    if n:
                        _rerun()
            st.divider()
            dfm = metas_do_dia(date.today())
            if dfm.empty:
                st.info("Nenhuma meta para hoje. Últimas metas:")
                dfm_last = metas_ultimas(10)
                st.dataframe(dfm_last, use_container_width=True, height=260) if not dfm_last.empty else st.caption("Sem histórico.")
            else:
                st.dataframe(dfm, use_container_width=True, height=260)

        # ---------- Ferramentas ----------
        with t_tools:
            st.subheader("Ferramentas administrativas")
            st.warning("⚠️ Apaga todos os dados. Use com cautela.", icon="⚠️")
            colA, colB = st.columns([2, 1])
            with colA:
                confirm = st.text_input("Digite EXACTAMENTE: ZERAR", placeholder="ZERAR")
            with colB:
                if st.button("🗑️ Zerar tabelas", type="secondary"):
                    if (confirm or "").strip().upper() == "ZERAR":
                        st.success(f"Tabelas limpas. Registros afetados: {wipe_all()}.")
                        _rerun()
                    else:
                        st.error("Confirmação inválida. Digite 'ZERAR'.")

    # -------- TV --------
    else:
        if IDX == 0:
            grid(df, "Coletas & Reposições — pág. 1/3", 520)
        elif IDX == 1:
            grid(df_over, "Atrasados — pág. 2/3", 560)
        else:
            grid(df_next7, "Próx. 7 dias — pág. 3/3", 560)
        st.caption(f"Tela {IDX + 1}/3 — alterna a cada {INTERVAL}s")
        components.html(
            f"""
        <script>
          setTimeout(function(){{
            const p=new URLSearchParams(window.location.search);
            p.set('tv','1'); p.set('interval','{INTERVAL}'); p.set('tv_page','{(IDX+1)%3}');
            location.replace(window.location.pathname+'?'+p.toString());
          }},{INTERVAL*1000});
        </script>""",
            height=0,
            width=0,
        )
