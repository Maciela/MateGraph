import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sympy import Piecewise, And, Or, solve, Eq, S, Union, Intersection, Interval, FiniteSet, nsimplify, limit
from sympy.abc import x
from sympy.calculus.util import continuous_domain, singularities
from sympy.core.relational import Relational
import warnings
warnings.filterwarnings("ignore")

# ---------- utilidades ----------
def _fmt_sym(v):
    try:
        if v == sp.oo: return "∞"
        if v == -sp.oo: return "-∞"
        return sp.sstr(v)
    except Exception:
        try:
            return f"{float(v):g}"
        except Exception:
            return str(v)

def formato_dominio(D):
    """Formatea conjuntos Sympy a texto legible."""
    try:
        if D == S.Reals:
            return "ℝ"
        if isinstance(D, Interval):
            left = "-∞" if D.left == -sp.oo else _fmt_sym(D.left)
            right = "∞" if D.right == sp.oo else _fmt_sym(D.right)
            lb = "(" if D.left_open else "["
            rb = ")" if D.right_open else "]"
            return f"{lb}{left}, {right}{rb}"
        if isinstance(D, Union):
            partes = []
            for a in D.args:
                partes.append(formato_dominio(a))
            return " ∪ ".join(partes)
        if isinstance(D, FiniteSet):
            pts = ", ".join(_fmt_sym(p) for p in sorted(list(D), key=lambda t: float(sp.N(t))))
            return "{" + pts + "}"
        return "{" + sp.sstr(D) + "}"
    except Exception:
        return sp.sstr(D)

def extract_breakpoints_from_condition(cond):
    """Extrae puntos críticos de las condiciones de funciones por partes"""
    points = set()
    try:
        solns = sp.solve(cond, x)
        if isinstance(solns, (list, tuple, set)):
            for s in solns:
                if s.is_real:
                     try:
                         points.add(float(sp.N(s)))
                     except:
                         pass
        elif hasattr(solns, 'args'):
             for arg in sp.preorder_traversal(solns):
                 if arg.is_real and arg.is_number:
                     try:
                         points.add(float(sp.N(arg)))
                     except:
                         pass
        elif solns.is_real and solns.is_number:
            try:
                points.add(float(sp.N(solns)))
            except:
                pass

        for arg in sp.preorder_traversal(cond):
            if isinstance(arg, (sp.StrictGreaterThan, sp.StrictLessThan, sp.GreaterThan, sp.LessThan, sp.Eq, sp.Ne)):
                lhs, rhs = arg.lhs, arg.rhs
                if lhs == x and rhs.is_number:
                     try:
                         points.add(float(sp.N(rhs)))
                     except:
                         pass
                elif rhs == x and lhs.is_number:
                     try:
                         points.add(float(sp.N(lhs)))
                     except:
                         pass

    except Exception:
        pass
    return sorted(list(points))

def evaluate_condition_at_point(condition, point_value):
    """Evalúa una condición booleana en un punto específico de manera robusta"""
    try:
        if condition == True:
            return True
        if condition == False:
            return False

        if isinstance(condition, And):
            return all(evaluate_condition_at_point(arg, point_value) for arg in condition.args)
        elif isinstance(condition, Or):
            return any(evaluate_condition_at_point(arg, point_value) for arg in condition.args)

        try:
            point_value_hp = sp.Float(point_value, dps=50)
            cond_substituted = condition.subs(x, point_value_hp)

            if isinstance(cond_substituted, (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan, sp.GreaterThan, sp.Eq, sp.Ne)):
                 try:
                    lhs = float(cond_substituted.lhs.evalf())
                    rhs = float(cond_substituted.rhs.evalf())

                    plot_tolerance = 1e-8

                    if isinstance(cond_substituted, sp.StrictLessThan):
                        return lhs < rhs - plot_tolerance
                    elif isinstance(cond_substituted, sp.LessThan):
                        return lhs <= rhs + plot_tolerance
                    elif isinstance(cond_substituted, sp.StrictGreaterThan):
                        return lhs > rhs + plot_tolerance
                    elif isinstance(cond_substituted, sp.GreaterThan):
                        return lhs >= rhs - plot_tolerance
                    elif isinstance(cond_substituted, sp.Eq):
                        return abs(lhs - rhs) < plot_tolerance
                    elif isinstance(cond_substituted, sp.Ne):
                        return abs(lhs - rhs) >= plot_tolerance
                 except:
                     pass

            try:
                 return bool(cond_substituted)
            except:
                 pass

        except Exception:
            pass

        return False

    except Exception:
        return False

def label_once(name, used):
    if name in used:
        return "_nolegend_"
    used.add(name)
    return name

# ---------- analysis utilities ----------
def safe_is_polynomial(f):
    try:
        sp.Poly(f, x)
        return True
    except Exception:
        return False

def analyze_single_expr(expr):
    is_poly = safe_is_polynomial(expr)
    is_rational = False
    has_exp = False
    has_log = False
    has_trig = False

    num, den = expr.as_numer_denom()
    if den != 1:
        is_rational = safe_is_polynomial(num) and safe_is_polynomial(den)

    for arg in sp.preorder_traversal(expr):
        if isinstance(arg, sp.exp):
            has_exp = True
        elif isinstance(arg, sp.Pow) and arg.exp.has(x):
            has_exp = True
        elif isinstance(arg, sp.log):
            has_log = True
        elif isinstance(arg, (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)):
            has_trig = True

    return {
        'is_poly': is_poly,
        'is_rational': is_rational,
        'has_exp': has_exp,
        'has_log': has_log,
        'has_trig': has_trig
    }

def detectar_tipo(f):
    if isinstance(f, Piecewise):
        return "Por partes"

    info = analyze_single_expr(f)
    if info['is_poly']:
        return "Polinómica"
    elif info['is_rational']:
        return "Racional"
    elif info['has_exp']:
        return "Exponencial"
    elif info['has_log']:
        return "Logarítmica"
    elif info['has_trig']:
        return "Trigonométrica"
    else:
        return "Otro"

# ---------- graficado mejorado para funciones por partes ----------
def plot_piecewise_simple(f, x_min, x_max, ax, used):
    """Grafica funciones por partes generando points directamente en los intervalos válidos"""
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    explicit_doms = []
    default_i = -1
    for ii, (_, cond) in enumerate(f.args):
        if cond == True:
            default_i = ii
            break
        else:
            try:
                d = sp.solveset(cond, x, domain=S.Reals)
                explicit_doms.append(d)
            except:
                explicit_doms.append(S.EmptySet)
    if default_i >= 0:
        union_explicit = Union(*explicit_doms)
        complement_dom = S.Reals - union_explicit
    else:
        complement_dom = S.EmptySet

    all_breakpoints = set()
    for expr_branch, cond in f.args:
        bps = extract_breakpoints_from_condition(cond)
        for bp in bps:
            if x_min - 1e-9 <= bp <= x_max + 1e-9:
                all_breakpoints.add(bp)

    all_breakpoints = sorted(list(all_breakpoints))
    points_to_mark = []
    plot_interval = Interval(x_min, x_max, left_open=False, right_open=False)

    for i, (expr_branch, cond) in enumerate(f.args):
        try:
            color = colors[i % len(colors)]

            if cond == True and default_i == i:
                domain_branch_cond = complement_dom
            else:
                try:
                    domain_branch_cond = sp.solveset(cond, x, domain=S.Reals)
                except:
                    domain_branch_cond = S.Reals

            domain_branch_expr = continuous_domain(expr_branch, x, S.Reals)
            valid_domain = Intersection(domain_branch_cond, domain_branch_expr, plot_interval)

            if valid_domain.is_EmptySet:
                continue

            if isinstance(valid_domain, Interval):
                intervals_or_points = [valid_domain]
            elif isinstance(valid_domain, Union):
                intervals_or_points = sorted(list(valid_domain.args), key=lambda item: float(sp.N(item.start)) if hasattr(item, 'start') and isinstance(item, Interval) else float(sp.N(item)))
            elif isinstance(valid_domain, FiniteSet):
                intervals_or_points = sorted(list(valid_domain), key=lambda t: float(sp.N(t)))
            else:
                 x_test = np.linspace(x_min, x_max, 5000)
                 x_plot = []
                 y_plot = []

                 for xi in x_test:
                    try:
                        xi_float = float(xi)
                        if evaluate_condition_at_point(cond, sp.Float(xi_float)):
                            yi = float(expr_branch.subs(x, sp.Float(xi_float)).evalf())
                            if np.isfinite(yi):
                                x_plot.append(xi_float)
                                y_plot.append(yi)
                    except:
                        pass

                 if len(x_plot) > 0:
                     ax.plot(x_plot, y_plot, color=color, linewidth=2.5,
                            label=label_once(f"Rama {i+1}: {sp.pretty(expr_branch)}", used))

                 fallback_breakpoints = [bp for bp in all_breakpoints if x_min - 1e-9 <= bp <= x_max + 1e-9]
                 for bp in fallback_breakpoints:
                     try:
                         y_at_bp = float(expr_branch.subs(x, sp.Float(bp)).evalf())
                         if np.isfinite(y_at_bp):
                              is_included = evaluate_condition_at_point(cond, sp.Float(bp))
                              points_to_mark.append((bp, y_at_bp, color, is_included))
                     except:
                          pass

                 continue

            first_segment = True
            for item in intervals_or_points:
                try:
                    if isinstance(item, Interval):
                        start_sym = item.start
                        end_sym = item.end

                        start_float = float(sp.N(start_sym)) if start_sym != -sp.oo else x_min
                        end_float = float(sp.N(end_sym)) if end_sym != sp.oo else x_max

                        plot_start = start_float + 1e-9 if item.left_open and start_sym != -sp.oo else start_float
                        plot_end = end_float - 1e-9 if item.right_open and end_sym != sp.oo else end_float

                        if plot_start > plot_end:
                             continue

                        num_points = max(2, int((plot_end - plot_start) / (x_max - x_min) * 5000))
                        x_vals = np.linspace(plot_start, plot_end, num_points)

                        branch_sing = [float(sp.N(s)) for s in singularities(expr_branch, x) if s.is_real]

                        y_vals = []
                        for xi in x_vals:
                            if any(abs(xi - s) < 1e-4 for s in branch_sing):
                                y_vals.append(np.nan)
                            else:
                                try:
                                    yi = float(expr_branch.subs(x, sp.Float(xi)).evalf())
                                    if np.isfinite(yi):
                                        y_vals.append(yi)
                                    else:
                                        y_vals.append(np.nan)
                                except:
                                    y_vals.append(np.nan)

                        x_vals = np.array(x_vals)
                        y_vals = np.array(y_vals)

                        if len(x_vals) > 0 and np.any(np.isfinite(y_vals)):
                             label = label_once(f"Rama {i+1}: {sp.pretty(expr_branch)}", used) if first_segment else "_nolegend_"
                             ax.plot(x_vals, y_vals, color=color, linewidth=2.5, label=label)
                             first_segment = False

                        if start_sym != -sp.oo:
                            try:
                                y_start_sym = expr_branch.subs(x, start_sym)
                                y_start = float(sp.N(y_start_sym))
                                if np.isfinite(y_start):
                                    is_included = not item.left_open
                                    points_to_mark.append((start_float, y_start, color, is_included))
                            except:
                                pass

                        if end_sym != sp.oo:
                             try:
                                 y_end_sym = expr_branch.subs(x, end_sym)
                                 y_end = float(sp.N(y_end_sym))
                                 if np.isfinite(y_end):
                                     is_included = not item.right_open
                                     points_to_mark.append((end_float, y_end, color, is_included))
                             except:
                                 pass

                    elif hasattr(item, 'is_number') and item.is_number:
                        try:
                            pt_float = float(sp.N(item))
                            y_pt = float(expr_branch.subs(x, sp.Float(pt_float)).evalf())
                            if np.isfinite(y_pt):
                                 points_to_mark.append((pt_float, y_pt, color, True))
                        except:
                            pass

                except Exception as e:
                     continue

        except Exception as e:
            continue

    marker_style = 'o'
    markersize = 8
    markeredgewidth = 2
    for px, py, pcolor, is_included in points_to_mark:
        ax.plot(px, py, marker=marker_style, markersize=markersize, zorder=10,
               markerfacecolor=pcolor if is_included else 'white',
               markeredgecolor=pcolor,
               markeredgewidth=1.5 if is_included else 2)

def plot_regular_function(f, x_min, x_max, ax, used):
    """Grafica funciones regulares (no por partes) con manejo mejorado de discontinuidades"""
    try:
        f_num = sp.lambdify(x, f, "numpy")

        sing = []
        try:
            sing_sympy = singularities(f, x)
            for s in sing_sympy:
                if s.is_real:
                    try:
                        sing.append(float(sp.N(s)))
                    except:
                        pass
        except:
            pass

        segments = []
        
        if not sing:
            segments = [(x_min, x_max)]
        else:
            sing_sorted = sorted([s for s in sing if x_min < s < x_max])
            
            current = x_min
            exclusion_radius = 0.01
            
            for s in sing_sorted:
                if current < s - exclusion_radius:
                    segments.append((current, s - exclusion_radius))
                current = s + exclusion_radius
            
            if current < x_max:
                segments.append((current, x_max))

        first_segment = True
        all_Y = []
        
        for seg_start, seg_end in segments:
            if seg_end - seg_start < 1e-6:
                continue
                
            num_points = max(100, int((seg_end - seg_start) / (x_max - x_min) * 2000))
            X_seg = np.linspace(seg_start, seg_end, num_points)
            Y_seg = []

            for xi in X_seg:
                try:
                    yi = f_num(xi)
                    if np.isscalar(yi) and np.isfinite(yi) and abs(yi) < 1e10:
                        Y_seg.append(float(yi))
                    else:
                        Y_seg.append(np.nan)
                except:
                    Y_seg.append(np.nan)

            X_seg = np.array(X_seg)
            Y_seg = np.array(Y_seg)
            
            valid_mask = np.isfinite(Y_seg)
            if np.any(valid_mask):
                X_valid = X_seg[valid_mask]
                Y_valid = Y_seg[valid_mask]
                
                if len(Y_valid) > 1:
                    y_diffs = np.abs(np.diff(Y_valid))
                    median_diff = np.median(y_diffs) if len(y_diffs) > 0 else 0
                    
                    large_jump_indices = np.where(y_diffs > max(10 * median_diff, 100))[0]
                    
                    if len(large_jump_indices) > 0:
                        start_idx = 0
                        for jump_idx in large_jump_indices:
                            if jump_idx + 1 - start_idx > 1:
                                label = label_once("f(x)", used) if first_segment else "_nolegend_"
                                ax.plot(X_valid[start_idx:jump_idx+1], Y_valid[start_idx:jump_idx+1], 
                                       color="#1f77b4", linewidth=2.5, label=label)
                                first_segment = False
                            start_idx = jump_idx + 1
                        
                        if len(X_valid) - start_idx > 1:
                            label = label_once("f(x)", used) if first_segment else "_nolegend_"
                            ax.plot(X_valid[start_idx:], Y_valid[start_idx:], 
                                   color="#1f77b4", linewidth=2.5, label=label)
                            first_segment = False
                    else:
                        if len(X_valid) > 1:
                            label = label_once("f(x)", used) if first_segment else "_nolegend_"
                            ax.plot(X_valid, Y_valid, color="#1f77b4", linewidth=2.5, label=label)
                            first_segment = False
                
                all_Y.extend(Y_valid)

        return np.array(all_Y)

    except Exception as e:
        return np.array([])

# ---------- análisis principal ----------
def analizar_funcion(tipo_ingresado, expr_text):
    expr_text = (expr_text or "").strip()
    if expr_text == "":
        return "Introduce una expresión válida.", None

    expr_text_clean = expr_text.replace("^", "**")
    expr_text_clean = expr_text_clean.replace(" e**", " E**").replace("(e**", "(E**")

    try:
        f = sp.sympify(expr_text_clean, evaluate=True)
    except Exception as e:
        return f"Error al interpretar la expresión: {e}", None

    is_piecewise = isinstance(f, Piecewise)
    tipo_detectado = detectar_tipo(f)

    if is_piecewise:
        branch_types = []
        for expr_branch, cond in f.args:
            info = analyze_single_expr(expr_branch)
            branch_types.append(info)

        if any(b['is_poly'] for b in branch_types):
            tipo_detectado += " (Polinómica)"
        elif any(b['is_rational'] for b in branch_types):
            tipo_detectado += " (Racional)"
        elif any(b['has_exp'] for b in branch_types):
            tipo_detectado += " (Exponencial)"
        elif any(b['has_log'] for b in branch_types):
            tipo_detectado += " (Logarítmica)"
        elif any(b['has_trig'] for b in branch_types):
            tipo_detectado += " (Trigonométrica)"

    # Calcular dominio
    try:
        if is_piecewise:
            dominios = []
            for expr_branch, cond in f.args:
                try:
                    if cond == True:
                        dom_cond = S.Reals
                    else:
                        try:
                            dom_cond = sp.solveset(cond, x, domain=S.Reals)
                            if isinstance(dom_cond, FiniteSet):
                                dom_cond = Union(*[Interval(d, d) for d in dom_cond])
                        except:
                             dom_cond = S.Reals

                    dom_expr = continuous_domain(expr_branch, x, S.Reals)
                    dom_total = Intersection(dom_expr, dom_cond) if dom_cond != S.Reals else dom_expr
                    dominios.append(dom_total)
                except Exception as e:
                    dominios.append(S.Reals)

            dominio = Union(*dominios) if dominios else S.Reals
        else:
            dominio = continuous_domain(f, x, S.Reals)

        dominio_str = formato_dominio(dominio)
    except Exception as e:
        dominio_str = "No se pudo determinar"

    # Encontrar ceros
    ceros_reales = []
    try:
        if is_piecewise:
            sols = set()
            for expr_branch, cond in f.args:
                try:
                    branch_sols = solve(Eq(expr_branch, 0), x)
                    if not isinstance(branch_sols, (list, tuple)):
                        branch_sols = [branch_sols] if branch_sols is not None else []
                    for s in branch_sols:
                        if s.is_real and evaluate_condition_at_point(cond, s):
                            try:
                                f_at_s = float(f.subs(x, s).evalf())
                                if abs(f_at_s) < 1e-9:
                                     sols.add(nsimplify(s))
                            except:
                                pass

                except Exception as e:
                    pass
            ceros_reales = sorted(list(sols), key=lambda t: float(sp.N(t)))
        else:
            sols = solve(Eq(f, 0), x, domain=S.Reals)
            ceros_reales = [c for c in sols if c.is_real]
            ceros_reales = sorted(ceros_reales, key=lambda t: float(sp.N(t)))
    except Exception as e:
        ceros_reales = []

    # Análisis de asíntotas y discontinuidades
    AV, AH, AO = [], [], []
    disc_evitables = []

    candidate_points = set()

    try:
        sing = singularities(f, x)
        for s in sing:
            if s.is_real:
                try:
                    candidate_points.add(float(sp.N(s)))
                except:
                    pass
    except Exception:
        pass

    if is_piecewise:
        for expr_branch, cond in f.args:
            bps = extract_breakpoints_from_condition(cond)
            for bp in bps:
                candidate_points.add(bp)
        for expr_branch, cond in f.args:
            try:
                sing_branch = singularities(expr_branch, x)
                for s in sing_branch:
                    if s.is_real:
                        try:
                            candidate_points.add(float(sp.N(s)))
                        except:
                            pass
            except:
                pass

    candidate_points = sorted(list(candidate_points))

    for a in candidate_points:
        try:
            if not (isinstance(a, (int, float)) or (hasattr(a, 'is_real') and a.is_real)):
                continue

            a_sym = sp.Float(a) if isinstance(a, (int, float)) else a

            lim_l = limit(f, x, a_sym, dir='-')
            lim_r = limit(f, x, a_sym, dir='+')

            try:
                 f_at = f.subs(x, a_sym)
            except Exception:
                 f_at = None

            is_vertical_asymptote = False
            
            if lim_l in (sp.oo, -sp.oo) or lim_r in (sp.oo, -sp.oo):
                is_vertical_asymptote = True
            elif limit(f, x, a_sym) in (sp.oo, -sp.oo):
                is_vertical_asymptote = True
            
            if is_vertical_asymptote:
                if not (lim_l.is_real and lim_r.is_real and abs(float(sp.N(lim_l)) - float(sp.N(lim_r))) < 1e-9):
                    AV.append(a_sym)
            else:
                if lim_l.is_real and lim_r.is_real and abs(float(sp.N(lim_l)) - float(sp.N(lim_r))) > 1e-9:
                     pass
                elif (lim_l.is_real and lim_r.is_real and abs(float(sp.N(lim_l)) - float(sp.N(lim_r))) < 1e-9):
                    if f_at is None or not (hasattr(f_at, 'is_real') and f_at.is_real) or (f_at.is_real and abs(float(sp.N(f_at)) - float(sp.N(lim_l))) > 1e-9):
                         disc_evitables.append((a_sym, lim_l))

        except Exception as e:
            pass

    # Asíntotas horizontales
    try:
        Lp = limit(f, x, sp.oo)
        Lm = limit(f, x, -sp.oo)
        if Lp.is_real and Lp not in (sp.oo, -sp.oo):
            if not any(abs(float(Lp) - float(ah)) < 1e-9 for ah in AH):
                 AH.append(Lp)
        if Lm.is_real and Lm not in (sp.oo, -sp.oo) and not any(abs(float(Lm) - float(ah)) < 1e-9 for ah in AH):
             AH.append(Lm)
    except Exception as e:
        pass

    # Asíntotas oblicuas
    try:
        m_try_p = limit(f/x, x, sp.oo)
        if m_try_p.is_real and abs(float(m_try_p)) > 1e-9:
            b_try_p = limit(f - m_try_p*x, x, sp.oo)
            if b_try_p.is_real:
                if not any(abs(float(m_try_p) - float(m)) < 1e-9 and abs(float(b_try_p) - float(b)) < 1e-9 for m, b in AO):
                    AO.append((m_try_p, b_try_p))

        m_try_m = limit(f/x, x, -sp.oo)
        if m_try_m.is_real and abs(float(m_try_m)) > 1e-9 and not any(abs(float(m_try_m) - float(m)) < 1e-9 for m, b in AO):
             b_try_m = limit(f - m_try_m*x, x, -sp.oo)
             if b_try_m.is_real:
                 if not any(abs(float(m_try_m) - float(m)) < 1e-9 and abs(float(b_try_m) - float(b)) < 1e-9 for m, b in AO):
                     AO.append((m_try_m, b_try_m))
    except Exception as e:
        pass

    # ---------- Crear gráfico ----------
    fig, ax = plt.subplots(figsize=(12, 8))
    used = set()

    # Determinar rango de graficado
    x_min_plot, x_max_plot = -10, 10
    if AV:
        try:
            avvals = [float(sp.N(v)) for v in AV]
            min_av = min(avvals)
            max_av = max(avvals)
            if min_av - 5 < x_min_plot:
                x_min_plot = min_av - 5
            if max_av + 5 > x_max_plot:
                x_max_plot = max_av + 5
        except:
            pass

    all_breakpoints_float = []
    if is_piecewise:
        bps_set = set()
        for _, cond in f.args:
            bps = extract_breakpoints_from_condition(cond)
            bps_set.update(bps)
        all_breakpoints_float = sorted(list(bps_set))
    if all_breakpoints_float:
        try:
            min_bp = min(all_breakpoints_float)
            max_bp = max(all_breakpoints_float)
            if min_bp - 2 < x_min_plot:
                x_min_plot = min_bp - 2
            if max_bp + 2 > x_max_plot:
                x_max_plot = max_bp + 2
        except:
            pass

    # Graficar función
    Y_data = None
    if is_piecewise:
        plot_piecewise_simple(f, x_min_plot, x_max_plot, ax, used)
    else:
        Y_data = plot_regular_function(f, x_min_plot, x_max_plot, ax, used)

    # Colores para elementos
    color_cero = "#ff7f0e"
    color_av = "#d62728"
    color_ah = "#2ca02c"
    color_ao = "#ff8c00"
    color_de = "#9467bd"

    # Graficar ceros
    for z in ceros_reales:
        try:
            zv = float(sp.N(z))
            if x_min_plot <= zv <= x_max_plot:
                 try:
                     f_at_z = float(f.subs(x, z).evalf())
                     if abs(f_at_z) < 1e-9:
                        ax.plot(zv, 0.0, marker='o', markersize=8, color=color_cero,
                               markerfacecolor=color_cero, markeredgecolor='white', markeredgewidth=1,
                               label=label_once("Cero", used), zorder=10)
                 except:
                     pass

        except Exception as e:
            pass

    # Graficar asíntotas verticales
    for a in AV:
        try:
            a_v = float(sp.N(a))
            if x_min_plot - 1e-9 <= a_v <= x_max_plot + 1e-9:
                ax.axvline(a_v, linestyle='--', color=color_av, linewidth=2, alpha=0.8,
                          label=label_once("Asíntota vertical", used), zorder=5)
        except Exception as e:
            pass

    # Graficar asíntotas horizontales
    for L in AH:
        try:
            Lf = float(sp.N(L))
            ax.axhline(Lf, linestyle='--', color=color_ah, linewidth=2, alpha=0.8,
                      label=label_once("Asíntota horizontal", used), zorder=5)
        except Exception as e:
            pass

    # Graficar asíntotas oblicuas
    for m_, b_ in AO:
        try:
            m_f = float(sp.N(m_))
            b_f = float(sp.N(b_))
            Xo = np.array([x_min_plot, x_max_plot])
            Yo = m_f * Xo + b_f
            ax.plot(Xo, Yo, linestyle='--', color=color_ao, linewidth=2, alpha=0.8,
                   label=label_once("Asíntota oblicua", used), zorder=5)
        except Exception as e:
            pass

    # Graficar discontinuidades evitables
    for a, L in disc_evitables:
        try:
            a_v = float(sp.N(a))
            L_v = float(sp.N(L))
            if x_min_plot <= a_v <= x_max_plot:
                ax.plot(a_v, L_v, marker='o', markersize=10, markerfacecolor='white',
                       markeredgecolor=color_de, markeredgewidth=2, zorder=10,
                       label=label_once("Discontinuidad evitable", used))
        except Exception as e:
            pass

    # Ejes principales
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.7)

    # Ajustar límites del gráfico
    y_min_plot, y_max_plot = -10, 10
    if Y_data is not None and Y_data.size > 0:
        finite_y = Y_data[np.isfinite(Y_data)]
        if finite_y.size > 0:
            p5, p95 = np.percentile(finite_y, [5, 95])
            rng = p95 - p5
            if rng > 0 and rng < 1e6:
                y_min_plot_data = p5 - 0.2 * rng
                y_max_plot_data = p95 + 0.2 * rng
            else:
                y_min_plot_data = np.min(finite_y) - 1
                y_max_plot_data = np.max(finite_y) + 1

            y_min_plot = min(y_min_plot, y_min_plot_data)
            y_max_plot = max(y_max_plot, y_max_plot_data)

    if AH:
        try:
            ah_vals = [float(sp.N(h)) for h in AH if hasattr(sp.N(h), 'is_real') and sp.N(h).is_real and abs(float(sp.N(h))) < 100]
            if ah_vals:
                 min_ah = min(ah_vals)
                 max_ah = max(ah_vals)
                 if min_ah - 2 < y_min_plot:
                    y_min_plot = min_ah - 2
                 if max_ah + 2 > y_max_plot:
                    y_max_plot = max_ah + 2
        except:
            pass

    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title(f"Gráfico de f(x) = {expr_text}", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)

    # Leyenda
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Crear texto de análisis
    ceros_txt = ", ".join(_fmt_sym(c) for c in ceros_reales) if ceros_reales else "Ninguno"
    av_txt = ", ".join(f"x = {_fmt_sym(a)}" for a in AV) if AV else "Ninguna"
    ah_txt = ", ".join(f"y = {_fmt_sym(L)}" for L in AH) if AH else "Ninguna"
    ao_txt = ", ".join(f"y = {_fmt_sym(m)}·x + {_fmt_sym(b)}" for m, b in AO) if AO else "Ninguna"
    de_txt = ", ".join(f"({_fmt_sym(a)}, {_fmt_sym(L)})" for a, L in disc_evitables) if disc_evitables else "Ninguna"

    analisis = (
        f"📋 ANÁLISIS DE LA FUNCIÓN\n"
        f"{'='*40}\n"
        f"🔸 Tipo ingresado: {tipo_ingresado}\n"
        f"🔸 Tipo detectado: {tipo_detectado}\n"
        f"🔸 Dominio: {dominio_str}\n\n"
        f"🎯 PUNTOS CARACTERÍSTICOS\n"
        f"{'='*40}\n"
        f"🔸 Ceros reales: {ceros_txt}\n\n"
        f"📈 ASÍNTOTAS\n"
        f"{'='*40}\n"
        f"🔸 Verticales: {av_txt}\n"
        f"🔸 Horizontales: {ah_txt}\n"
        f"🔸 Oblicuas: {ao_txt}\n\n"
        f"⚠️  DISCONTINUIDADES\n"
        f"{'='*40}\n"
        f"🔸 Evitables: {de_txt}\n"
    )

    return analisis, fig

# ---------- Interfaz Gradio ----------
ejemplos_funciones = [
    ["Polinómica", "x**2 - 4"],
    ["Polinómica", "x**3 - 2*x + 1"],
    ["Racional", "(x+1)/(x-2)"],
    ["Racional", "(2*x**2 + 3)/(x**2 - 1)"],
    ["Exponencial", "exp(x)"],
    ["Exponencial", "2**x"],
    ["Logarítmica", "log(x)"],
    ["Trigonométrica", "sin(x)"],
    ["Trigonométrica", "cos(x)"],
    ["Por partes", "Piecewise((x**2, x<0), (x, x>=0))"]
]

iface = gr.Interface(
    fn=analizar_funcion,
    inputs=[
        gr.Dropdown(
            choices=['Polinómica', 'Racional', 'Exponencial', 'Logarítmica', 'Trigonométrica', 'Por partes'],
            value='Por partes',
            label='🎯 Tipo de función'
        ),
        gr.Textbox(
            label='✏️ Expresión de f(x)',
            value='Piecewise((x**2, x<1), (2*x-1, x>=1))',
            placeholder='Escribe tu función aquí...',
            lines=2
        )
    ],
    outputs=[
        gr.Textbox(label='📊 Análisis detallado', lines=15),
        gr.Plot(label='📈 Gráfico de la función')
    ],
    title='🧮 Graficador de Funciones Reales',
    description="""
    **🎯 Analiza y grafica funciones de variable real**

    **📝 Sintaxis para funciones por partes:**

    Piecewise((expresión1, condición1), (expresión2, condición2), ...)

    **🔧 Operadores disponibles:**
    - Potencias: `**` (ej: `x**2`)
    - Exponencial: `exp(x)` o `E**x`
    - Logaritmo: `log(x)`
    - Trigonométricas: `sin(x)`, `cos(x)`, `tan(x)`
    - Raíz cuadrada: `sqrt(x)`

    **⚡ Condiciones para funciones por partes:**
    - `x < 1`, `x <= 1`, `x > 1`, `x >= 1`
    - `x == 0`, `x != 0`
    - `(x > -1) & (x < 1)` para intervalos
    - `Eq(x, 3)` para valores puntuales
    """,
    examples=ejemplos_funciones,
    allow_flagging='never',
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch()