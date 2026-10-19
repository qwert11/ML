"""Пошаговая трассировка учебной программы для сайта.

stdin — исходный код. stdout — JSON {"steps": [...], "output": "..."}.
Шаг = один выполненный оператор: какие строки он занимает, в какой функции,
все переменные этой функции после него (округлены до 4 знаков, а в n — точно)
и сколько строк к этому моменту напечатано. Заходим только в функции из нескольких
операторов: однострочные помощники вроде sigmoid считаются за один шаг.
"""
import ast
import io
import json
import sys

FILE = '<mini>'
source = sys.stdin.read()
tree = ast.parse(source)
lines = source.splitlines()

def names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for t in target.elts for n in names(t)]
    return []


# Строка → (первая, последняя) строка оператора; у for/def — только заголовок.
# targets — чему оператор присваивает, loops/bodies — строки тела for и def.
span, skip, traced, targets, loops, bodies = {}, set(), {'<module>'}, {}, {}, {}
for node in ast.walk(tree):
    if not isinstance(node, ast.stmt):
        continue
    end = node.end_lineno
    if hasattr(node, 'body'):
        end = node.body[0].lineno - 1
        while end > node.lineno and lines[end - 1].strip()[:1] in ('', '#'):
            end -= 1
    stmt = (node.lineno, end)
    for n in range(node.lineno, end + 1):
        span[n] = stmt
    if isinstance(node, ast.FunctionDef):
        skip.add(stmt)
        if len(node.body) > 1:
            traced.add(node.name)
    elif isinstance(node, ast.Assign):
        targets[stmt] = [n for t in node.targets for n in names(t)]
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets[stmt] = names(node.target)
    elif isinstance(node, ast.For):
        targets[stmt] = names(node.target)
        loops[stmt] = (node.body[0].lineno, node.end_lineno)
    if isinstance(node, (ast.For, ast.FunctionDef)):
        bodies[node.lineno] = [node.body[0].lineno, node.end_lineno]


def fmt(v):
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, float):
        return repr(round(v, 4) + 0.0)  # + 0.0 убирает «-0.0»
    if isinstance(v, (int, str)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        items = [fmt(x) for x in v]
        if None in items:
            return None
        text = ', '.join(items)
        return f'[{text}]' if isinstance(v, list) else f'({text}{"," if len(v) == 1 else ""})'
    if isinstance(v, dict):
        items = [(fmt(k), fmt(x)) for k, x in v.items()]
        if any(None in pair for pair in items):
            return None
        return '{' + ', '.join(f'{k}: {x}' for k, x in items) + '}'
    return None  # модули, функции — не данные


def scope(frame):
    out = {}
    for name, value in frame.f_locals.items():
        if not name.startswith('_'):
            text = fmt(value)
            if text is not None:
                out[name] = text
    return out


out = io.StringIO()
steps, state = [], {}


def emit(frame, stmt, next_line=None, **extra):
    st = state[id(frame)]
    now = scope(frame)
    changed = [k for k, v in now.items() if st['vars'].get(k) != v]
    st['vars'] = now
    name = frame.f_code.co_name
    step = {'l': list(stmt), 'f': '' if name == '<module>' else name,
            'v': now, 'c': changed, 't': targets.get(stmt, []),
            'o': out.getvalue().count('\n'),
            # точные значения (копия: списки дальше меняются) — для формул при сборке
            'n': json.loads(json.dumps({k: frame.f_locals[k] for k in now}))}
    body = loops.get(stmt)
    if body and not (next_line and body[0] <= next_line <= body[1]):
        step['x'] = 1  # заголовок for отработал вхолостую: цикл окончен
        step['t'] = []
    step.update(extra)
    steps.append(step)


def tracer(frame, event, arg):
    code = frame.f_code
    if code.co_filename != FILE or code.co_name not in traced:
        return None
    key = id(frame)
    if event == 'call':
        state[key] = {'stmt': None, 'vars': {}}
        if code.co_name != '<module>':
            emit(frame, span[code.co_firstlineno], call=frame.f_back.f_lineno)
        return tracer
    st = state[key]
    if event == 'line':
        stmt = span[frame.f_lineno]
        if stmt != st['stmt']:
            if st['stmt'] is not None and st['stmt'] not in skip:
                emit(frame, st['stmt'], frame.f_lineno)
            st['stmt'] = stmt
    elif event == 'return':
        if st['stmt'] not in skip:
            extra = {} if code.co_name == '<module>' else {'r': fmt(arg) or repr(arg)}
            emit(frame, st['stmt'], **extra)
        del state[key]
    return tracer


compiled = compile(source, FILE, 'exec')
real_stdout, sys.stdout = sys.stdout, out
sys.settrace(tracer)
try:
    exec(compiled, {'__name__': '__main__'})
finally:
    sys.settrace(None)
    sys.stdout = real_stdout
print(json.dumps({'steps': steps, 'bodies': bodies, 'output': out.getvalue()}, ensure_ascii=False))
