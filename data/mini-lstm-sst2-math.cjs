// Формулы сети к шагам мини-программы (только сборка, в браузер уходит готовый LaTeX).
// На каждом шаге: система уравнений текущей части сети, подсвечено уравнение этой строки,
// под ней — подстановка чисел.
// Точность — как в программе: результат каждой формулы с 6 значащими цифрами (Python f"{x:.6g}",
// печатается как repr: 2.0, 0.544879), и дальше он используется округлённым — программа делает то же
// функцией r(). Результат на странице обязан совпасть с программой знак в знак, иначе сборка падает.
// Числа внутри одной формулы (e^{-z}, ln, произведения) программа не округляет; здесь они показаны
// с 10 значащими цифрами, каждое посчитано от показанных перед ним, и сборка проверяет, что пересчёт
// по ним даёт те же 6 цифр результата.
// Всё, что программа печатает как «имя = число», сверяется с числом страницы под тем же именем.

// Как Python печатает float (repr): кратчайшая запись, которая восстанавливает число; у целых хвост «.0»
// (2.0, 0.0); e-запись при |x| < 1e-4 или ≥ 1e16 (1e-07, 1e+16).
const pyRepr = v => {
  if (v === 0) return Object.is(v, -0) ? '-0.0' : '0.0';
  const neg = v < 0, [m, ex] = Math.abs(v).toExponential().split('e'), exp = Number(ex), digits = m.replace('.', '');
  let s;
  if (exp < -4 || exp >= 16) s = m + 'e' + (exp < 0 ? '-' : '+') + String(Math.abs(exp)).padStart(2, '0');
  else if (exp < 0) s = '0.' + '0'.repeat(-exp - 1) + digits;
  else if (digits.length <= exp + 1) s = digits.padEnd(exp + 1, '0') + '.0';
  else s = digits.slice(0, exp + 1) + '.' + digits.slice(exp + 1);
  return (neg ? '-' : '') + s;
};
// Число, округлённое до digits значащих цифр, как его напечатает Python после float(f"{x:.{digits}g}").
// Проверено на 7000 случайных чисел; единственное отличие — точные двоичные «половинки» вроде 2⁻¹⁰ = 0.0009765625
// (Python округляет к чётному, JS — вверх). У выходов сигмоиды, tanh и ln таких чисел не бывает, а сборка всё равно
// сверяет каждое число с программой.
const py = (x, digits) => pyRepr(Number(x.toPrecision(digits)));
const N = x => py(x, 6), R = x => Number(N(x));        // результат: как печатает программа
const N10 = x => py(x, 10), R10 = x => Number(N10(x)); // промежуточное число внутри формулы
const M = x => x < 0 ? `(${N(x)})` : N(x);              // отрицательное — в скобках
const M10 = x => x < 0 ? `(${N10(x)})` : N10(x);
const N4 = x => pyRepr(Number(x.toFixed(4)) + 0);       // round(x, 4) + 0.0 — строки «после обучения»
const pyList = (a, F = N) => Array.isArray(a) ? `[${a.map(x => pyList(x, F)).join(', ')}]` : F(a); // repr списка Python
const pair = v => `(${N(v[0])},\\ ${N(v[1])})`;
// Школьное округление десятичной записи до 6 значащих цифр: половина — вверх. Нужно там, где точная сумма
// может лечь ровно посередине (z — произведения коротких десятичных дробей): у float64 там шум в 17-м знаке,
// и сборка проверяет, что программа округлила так же, как сделал бы человек с калькулятором.
const halfUp6 = x => {
  const s = N10(x); if (s.includes('e')) return N(x);
  const neg = s.startsWith('-'), [ip, fp = ''] = (neg ? s.slice(1) : s).split('.');
  let m = BigInt(ip + fp), scale = fp.length; const len = m.toString().length;
  if (len > 6) { const drop = 10n ** BigInt(len - 6); m = (m + drop / 2n) / drop; scale -= len - 6; }
  let str = m.toString();
  if (scale > 0) { str = str.padStart(scale + 1, '0'); str = (str.slice(0, -scale) + '.' + str.slice(-scale)).replace(/\.?0+$/, ''); }
  else if (scale < 0) str += '0'.repeat(-scale);
  return (neg ? '-' : '') + (str.includes('.') ? str : str + '.0');
};
const sum = a => a.reduce((acc, x) => acc + x, 0); // как sum() в Python: слагаемое к слагаемому, от нуля
const aligned = rows => `\\begin{aligned}${rows.join('\\\\')}\\end{aligned}`;
const table = (cols, rows, F = N) => `\\begin{array}{r|${'r'.repeat(cols.length)}} & ${cols.join(' & ')}\\\\ \\hline ${rows.map(([name, vals]) => `${name} & ${vals.map(F).join(' & ')}`).join('\\\\')}\\end{array}`;

const GATES = [['i', 'i', '\\sigma'], ['f', 'f', '\\sigma'], ['c_hat', '\\tilde c', '\\tanh'], ['o', 'o', '\\sigma']];
const GATE_COLS = ['i', 'f', '\\tilde c', 'o'], GK = ['i', 'f', 'c', 'o'];
const col = sym => sym === '\\tilde c' ? 'c' : sym;
// Порядок как в коде и в Keras: вектор слова x_t умножается на столбец W_i таблицы W (x @ W).
const LSTM = [
  ['x', 'x_t = E[w_t]'],
  ...GATES.map(([key, sym, fn]) => [key, `${sym}_t = ${fn}(x_t W_${col(sym)} + h_{t-1} U_${col(sym)} + b_${col(sym)})`]),
  ['c', 'c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde c_t'],
  ['h', 'h_t = o_t \\odot \\tanh(c_t)']
];
// Конец эпохи: Dense, метрика, ошибка и начало обратного прохода. δ перед величиной — ∂L/∂(величина).
const DENSE = [
  ['p', 'p_k = \\sigma(w_d\\,h^{(k)} + b_d)'],
  ['acc', '\\mathrm{acc} = \\frac1N\\sum_k \\big[\\,[p_k > 0{.}5] = y_k\\,\\big]'],
  ['L', 'L = -\\frac1N\\sum_k \\big[y_k \\ln p_k + (1-y_k)\\ln(1-p_k)\\big]'],
  ['dz_d', '\\delta z^d_k = \\frac{\\partial L}{\\partial(w_d h^{(k)} + b_d)} = \\frac{p_k - y_k}{N}'],
  ['g_wd', 'g_{w_d} = \\sum_k \\delta z^d_k\\, h^{(k)}'],
  ['g_bd', 'g_{b_d} = \\sum_k \\delta z^d_k'],
  ['dh', '\\delta h^{(k)}_T = \\delta z^d_k\\, w_d']
];
// Обратный проход через ячейку для одного слова (от последнего слова к первому).
const BACK = [
  ['do', '\\delta o_t = \\delta h_t \\cdot \\tanh(c_t)'],
  ['dc', '\\delta c_t = \\delta c_t^{\\leftarrow} + \\delta h_t \\cdot o_t \\cdot (1 - \\tanh^2 c_t)'],
  ['di', '\\delta i_t = \\delta c_t \\cdot \\tilde c_t'],
  ['df', '\\delta f_t = \\delta c_t \\cdot c_{t-1}'],
  ['dc_hat', '\\delta \\tilde c_t = \\delta c_t \\cdot i_t'],
  ['dz_i', '\\delta z_i = \\delta i_t \\cdot i_t (1 - i_t)'],
  ['dz_f', '\\delta z_f = \\delta f_t \\cdot f_t (1 - f_t)'],
  ['dz_c', '\\delta z_c = \\delta \\tilde c_t \\cdot (1 - \\tilde c_t^{\\,2})'],
  ['dz_o', '\\delta z_o = \\delta o_t \\cdot o_t (1 - o_t)'],
  ['gW', 'g_{W_{jk}} \\mathrel{+}= x_{t,j}\\, \\delta z_k'],
  ['gU', 'g_{U_k} \\mathrel{+}= h_{t-1}\\, \\delta z_k'],
  ['gb', 'g_{b_k} \\mathrel{+}= \\delta z_k'],
  ['dx', '\\delta x_{t,j} = \\sum_k W_{jk}\\, \\delta z_k \\;\\to\\; g_{E_{w_t,j}} \\mathrel{+}= \\delta x_{t,j}'],
  ['dh_prev', '\\delta h_{t-1} = \\sum_k U_k\\, \\delta z_k'],
  ['dc_prev', '\\delta c_{t-1} = \\delta c_t \\cdot f_t']
];
// Шаг rmsprop — одинаков для каждого веса θ (E, W, U, b, w_d, b_d).
const UPDATE = [
  ['v', 'v \\leftarrow \\rho\\,v + (1-\\rho)\\,g^2'],
  ['theta', '\\theta \\leftarrow \\theta - \\eta\\,\\frac{g}{\\sqrt{v + \\varepsilon}}']
];
// F — как показать аргумент: N для z вентиля (его печатает программа), N10 для w_d·h + b_d (промежуточное).
const plain = name => name.replace(/\\text\{|\}/g, ''); // имя веса без 	ext{} в заголовке
const sigma = (z, e, val, F = N) => `\\sigma(${F(z)}) = \\frac{1}{1+e^{${F(0 - z)}}} = \\frac{1}{1+${N10(e)}} = ${N(val)}`;

module.exports = function addMath(steps, code, output) {
  const src = code.split('\n');
  let G = {}, review = 0, sentence = '', t = 0, epoch = 0, cell = {}, back = {}, upd = null;
  const S = { H: [] };       // показанные значения: из них считаются следующие формулы
  const P = {};              // имя → число, как его печатает программа: сверка печати со страницей
  let train = {}, bvals = {}; // значения уже посчитанных строк текущей системы
  let paramFull = [], wordFull = [], gateRows = [], denseFull = [], backFull = [], backSteps = [];
  const param = m => { paramFull.push(m.sub); return m; };
  const dn = m => { denseFull.push(m.sub); return m; };
  const bk = m => { backFull.push(m.sub); return m; };
  const check = (what, actual, exact, shown) => {
    const want = N(actual);
    if (N(exact) !== want) throw Error(`Формула ${what}: по формуле ${N(exact)}, в программе ${want}`);
    if (shown !== undefined && N(shown) !== want) throw Error(`Формула ${what}: по показанным промежуточным ${N(shown)}, в программе ${want}`);
    return actual;
  };
  const system = (rows, vals, cur) => rows.map(([key, tex]) => ({ t: tex + (vals[key] || ''), s: cur.includes(key) ? 'cur' : vals[key] ? 'done' : 'todo' }));
  const epochTitle = () => `model.fit · эпоха ${epoch}`;
  const lstmTitle = () => `${epochTitle()} → вперёд · отзыв ${review} «${sentence}»` + (t ? ` · t = ${t}, слово «${cell.word}»` : '');
  const backTitle = () => `${epochTitle()} → назад · отзыв ${review} «${sentence}»` + (back.word ? ` · t = ${back.t}, слово «${back.word}»` : '');
  const lstm = (cur, extra) => ({ title: lstmTitle(), rows: system(LSTM, cell.vals || {}, cur), ...extra });
  const bsys = (cur, extra) => ({ title: backTitle(), rows: system(BACK, bvals, cur), ...extra });
  const dense = (cur, extra) => ({ title: `${epochTitle()} → Dense и ошибка`, rows: system(DENSE, train, cur), ...extra });
  const printed = (title, rows, cap) => ({ title, rows, cap: cap || 'Строка печатает посчитанное: те же числа, что в формулах выше, — сверьте с панелью «Вывод программы».' });
  const hRow = () => `H = (${S.H.map(N).join(',\\ ')})`;
  const wordOf = idx => `\\text{${G.words[idx]}}`;

  // Все 28 обновлений rmsprop эпохи считаются разом из таблиц градиентов и старых весов (таблицы — страницы).
  function computeUpdate(g) {
    const rho = S.rho, lr = S.lr, eps = S.eps;
    const one = (theta, v, grad) => { const vn = R(rho * v + (1 - rho) * grad ** 2); return { v: vn, theta: R(theta - lr * grad / Math.sqrt(vn + eps)), g: grad, old: theta, vOld: v }; };
    const map2 = (T, V, Gr) => T.map((row, a) => row.map((x, b) => one(x, V[a][b], Gr[a][b])));
    const map1 = (T, V, Gr) => T.map((x, a) => one(x, V[a], Gr[a]));
    upd = { E: map2(g.E, g.vE, g.gE), W: map2(g.W, g.vW, g.gW), U: map1(g.U, g.vU, g.gU), b: map1(g.b, g.vb, g.gb), w_d: one(g.w_d, g.v_wd, g.g_wd), b_d: one(g.b_d, g.v_bd, g.g_bd) };
    P.vE = upd.E.map(r => r.map(u => u.v)); P.E = upd.E.map(r => r.map(u => u.theta));
    P.vW = upd.W.map(r => r.map(u => u.v)); P.W = upd.W.map(r => r.map(u => u.theta));
    P.vU = upd.U.map(u => u.v); P.U = upd.U.map(u => u.theta); P.vb = upd.b.map(u => u.v); P.b = upd.b.map(u => u.theta);
    P.v_wd = upd.w_d.v; P.w_d = upd.w_d.theta; P.v_bd = upd.b_d.v; P.b_d = upd.b_d.theta;
  }
  // Какой вес обновляет текущий вызов rms(): по строке вызова и индексам циклов на верхнем уровне.
  function updTarget(callLine) {
    const line = src[callLine - 1] || '';
    if (line.includes('E[w][j]')) return { name: `E_{${wordOf(G.w)},${G.j}}`, u: upd.E[G.w][G.j] };
    if (line.includes('W[j][k]')) return { name: `W_{${G.j},${GK[G.k]}}`, u: upd.W[G.j][G.k] };
    if (line.includes('U[k]')) return { name: `U_{${GK[G.k]}}`, u: upd.U[G.k] };
    if (line.includes('b[k]')) return { name: `b_{${GK[G.k]}}`, u: upd.b[G.k] };
    if (line.includes('w_d, v_wd')) return { name: 'w_d', u: upd.w_d };
    return { name: 'b_d', u: upd.b_d };
  }
  const updSub = (name, u) => aligned([
    `v_{${name}} &= ${N(S.rho)}\\cdot ${N(u.vOld)} + ${N(1 - S.rho)}\\cdot ${M(u.g)}^2 = ${N(S.rho)}\\cdot ${N(u.vOld)} + ${N(1 - S.rho)}\\cdot ${N10(u.g ** 2)} = ${N(u.v)}`,
    `${name} &= ${M(u.old)} - ${N(S.lr)}\\cdot\\frac{${M(u.g)}}{\\sqrt{${N(u.v)} + 10^{-7}}} = ${M(u.old)} - ${N(S.lr)}\\cdot ${M10(u.g / R10(Math.sqrt(u.v + S.eps)))} = ${N(u.theta)}`]);
  const updTables = () => {
    const rowsE = G.words.map((w, a) => [`\\text{${w}}`, [0, 1].map(b => upd.E[a][b])]);
    const tab = (cols, rows, pick) => table(cols, rows.map(([n, us]) => [n, us.map(pick)]));
    return [
      `g_E = ${tab(['j{=}0', 'j{=}1'], rowsE, u => u.g)},\\quad v_E = ${tab(['j{=}0', 'j{=}1'], rowsE, u => u.v)},\\quad E = ${tab(['j{=}0', 'j{=}1'], rowsE, u => u.theta)}`,
      `g_W = ${tab(GATE_COLS, [['x_{t,0}', upd.W[0]], ['x_{t,1}', upd.W[1]]], u => u.g)},\\quad v_W = ${tab(GATE_COLS, [['x_{t,0}', upd.W[0]], ['x_{t,1}', upd.W[1]]], u => u.v)},\\quad W = ${tab(GATE_COLS, [['x_{t,0}', upd.W[0]], ['x_{t,1}', upd.W[1]]], u => u.theta)}`,
      `g_U = ${tab(GATE_COLS, [['h_{t-1}', upd.U]], u => u.g)},\\quad v_U = ${tab(GATE_COLS, [['h_{t-1}', upd.U]], u => u.v)},\\quad U = ${tab(GATE_COLS, [['h_{t-1}', upd.U]], u => u.theta)}`,
      `g_b = ${tab(GATE_COLS, [['1', upd.b]], u => u.g)},\\quad v_b = ${tab(GATE_COLS, [['1', upd.b]], u => u.v)},\\quad b = ${tab(GATE_COLS, [['1', upd.b]], u => u.theta)}`,
      updSub('w_d', upd.w_d), updSub('b_d', upd.b_d)];
  };

  function math(line, s, v, g) {
    const at = prefix => line.startsWith(prefix);
    const inFn = s.f;
    if (at('print("train_pad:')) {
      s.k = { title: '1–3. Данные → номера', cap: 'Формул сети здесь нет: словарь, номер каждого слова и выравнивание нулями слева. Результат — в переменных: word2idx и train_pad.' };
      return null;
    }
    if (at('E = [[')) return param({ title: 'Embedding: слово → вектор', rows: system([LSTM[0]], {}, ['x']),
      sub: `E = ${table(['x_{t,0}', 'x_{t,1}'], g.E.map((row, k) => [`\\text{${g.words[k]}}`, row]))}`,
      cap: 'Обучаемая таблица: строка на слово. В книге в строке 32 числа, здесь 2. Числа — стартовые веса Keras (seed 0), скопированы из его вывода выше. В коде E[idx] — строка с номером слова, в формулах она же x_t.' });
    if (at('W = [[')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `W = ${table(GATE_COLS, [['x_{t,0}', g.W[0]], ['x_{t,1}', g.W[1]]])}`, cap: 'Веса от входа x_t: столбец на вентиль. W_i в формулах — столбец i этой таблицы. Стартовые веса Keras (glorot_uniform, seed 0).' });
    if (at('U = [')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `U = ${table(GATE_COLS, [['h_{t-1}', g.U]])}`, cap: 'Веса от выхода прошлого шага h_{t−1}: U_i — столбец i. Стартовые веса Keras (orthogonal, seed 0).' });
    if (at('b = [')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `b = ${table(GATE_COLS, [['1', g.b]])}`, cap: 'Смещения вентилей ячейки (b_i, b_f, b_c, b_o). У f смещение 1 — так Keras делает по умолчанию (unit_forget_bias).' });
    if (at('w_d, b_d = ')) { S.w = v.w_d; S.b = v.b_d; return param({ title: 'Dense: параметры', rows: system([DENSE[0]], {}, []), sub: `w_d = ${N(S.w)},\\quad b_d = ${N(S.b)}`, cap: 'Веса слоя Dense(1): один вес на h и смещение (стартовые веса Keras). Индекс d — Dense: b без индекса выше — смещения вентилей LSTM.' }); }
    if (at('lr, rho, eps = ')) {
      S.lr = v.lr; S.rho = v.rho; S.eps = v.eps;
      const m = param({ title: 'compile: параметры rmsprop', rows: system(UPDATE, {}, ['v', 'theta']), sub: `\\eta = ${N(S.lr)},\\quad \\rho = ${N(S.rho)},\\quad \\varepsilon = 10^{-7}`,
        cap: 'optimizer="rmsprop" — это ровно эти числа, значения Keras по умолчанию: lr в коде — η, rho — ρ, eps — ε. Формулы сработают для каждого веса θ в конце каждой эпохи.' });
      s.k = { title: '4–5. Модель и compile: все числа до обучения', rows: system(LSTM, {}, []).concat(system(DENSE, {}, []), system(UPDATE, {}, [])), full: paramFull,
        cap: 'Таблица Embedding, веса ячейки LSTM, Dense и параметры rmsprop — всё, с чем начинается fit. Веса — настоящие стартовые веса Keras с seed 0, ничего не подставлено. Формулы сверху — те, что дальше будут считаться по этим числам.' };
      return m;
    }
    if (at('vE = ')) return { title: 'model.fit · перед эпохами', rows: system(UPDATE, {}, []), sub: 'v_E = v_W = v_U = v_b = v_{w_d} = v_{b_d} = 0', cap: 'RMSprop копит средний квадрат градиента отдельно для каждого веса: vE, vW, vU, vb, v_wd, v_bd. В начале все нули.' };
    if (at('vW = ') || at('vU = ') || at('vb = ') || at('v_wd = v_bd')) return { title: 'model.fit · перед эпохами', rows: system(UPDATE, {}, []), cap: 'Нули под средний квадрат градиента — по числу на каждый вес.' };
    if (at('for epoch')) {
      if (s.x) return { title: 'model.fit', rows: [], cap: 'Эпохи кончились: fit завершён, все веса — итоговые.' };
      epoch = v.epoch; train = {}; S.H = []; denseFull = []; backSteps = []; upd = null;
      return { title: epochTitle(), rows: system(LSTM, {}, []).concat(system(DENSE, {}, []), system(BACK, {}, []), system(UPDATE, {}, [])), cap: 'Эпоха = прямой проход по обоим отзывам, ошибка, обратный проход (градиенты всех весов) и один шаг rmsprop на всём батче. Учатся все слои, как в книге.' };
    }
    if (at('print(f"Epoch')) return printed(epochTitle(), system(LSTM, {}, []), 'Заголовок эпохи — как у Keras. Дальше: вперёд по словам, Dense, ошибка, назад по словам, шаг rmsprop.');
    if (at('H, mem = ')) return { title: epochTitle(), rows: system(LSTM, {}, []), cap: 'H — выход LSTM по каждому отзыву, mem — всё, что ячейка посчитала на каждом слове: пригодится обратному проходу.' };

    // ── прямой проход ──
    if (at('for k, seq')) {
      if (s.x) return { title: `${epochTitle()} → выход LSTM`, rows: [{ t: hRow(), s: 'cur' }], cap: 'Отзывы кончились: H собран, по одному числу на отзыв. Дальше — Dense.' };
      review = v.k + 1; t = 0; cell = {}; sentence = g.sentences[v.k];
      return lstm([], { cap: 'Новый отзыв: его слова по очереди пройдут через одну и ту же ячейку.' });
    }
    if (at('print(f" отзыв {k + 1} «{sentences[k]}»: вперёд')) return lstm([], { cap: 'Заголовок отзыва в выводе. Память между отзывами не передаётся: ниже h и c снова станут нулями.' });
    if (at('h, c, steps = 0.0')) { S.h = 0; S.c = 0; return lstm([], { sub: 'h_0 = 0,\\quad c_0 = 0', cap: 'Перед первым словом памяти нет; steps — список для обратного прохода.' }); }
    if (at('for idx in seq')) {
      if (s.x) return { title: `${epochTitle()} → выход LSTM · отзыв ${review}`, rows: [{ t: `h_T = h_${t} = ${N(S.h)}`, s: 'cur' }], cap: 'Слова кончились. Классификатор получит только последний выход h_T.' };
      t++; cell = { word: g.words[v.idx], idx: v.idx, vals: {} };
      return lstm([], { cap: `Слово ${t} из ${g.max_sen_len}: «${cell.word}», его номер ${v.idx}.` });
    }
    if (at('x = E[idx]') && !inFn) {
      const x = g.E[cell.idx]; cell.x = x; P.x = x; cell.vals.x = ` = ${pair(x)}`;
      return lstm(['x'], { sub: `x_${t} = E[\\text{${cell.word}}] = E_{${cell.idx}} = ${pair(x)}`, cap: 'Embedding: номер слова выбирает строку таблицы E.' });
    }
    if (at('print(f"{words[idx]:>5}: x = ')) {
      P.h = S.h; P.c = S.c;
      return lstm(['x'], { sub: `x_${t} = ${pair(cell.x)},\\quad h_{${t - 1}} = ${N(S.h)},\\quad c_{${t - 1}} = ${N(S.c)}`, cap: 'Вход шага: вектор слова и память после прошлого слова. Дальше считает lstm_step.' });
    }
    if (at('def lstm_step')) {
      cell.hp = check('h_{t-1}', v.h, S.h); cell.cp = check('c_{t-1}', v.c, S.c);
      wordFull = [`x_${t} = E[\\text{${cell.word}}] = ${pair(cell.x)},\\quad h_{${t - 1}} = ${N(cell.hp)},\\quad c_{${t - 1}} = ${N(cell.cp)}`]; gateRows = [];
      return lstm([], { sub: `x_t = ${pair(cell.x)},\\quad h_{t-1} = ${N(cell.hp)},\\quad c_{t-1} = ${N(cell.cp)}`, cap: 'Вызов функции: вектор слова и память прошлого шага стали её x, h, c.' });
    }
    if (at('z = [r(')) {
      const { W, U, b } = g, x = cell.x;
      cell.z = GATES.map(([key], k) => {
        const exact = x[0] * W[0][k] + x[1] * W[1][k] + cell.hp * U[k] + b[k], z = check(`z_${key}`, v.z[k], exact);
        if (halfUp6(exact) !== N(z)) throw Error(`Формула z_${key}: точно ${N10(exact)}, вверх — ${halfUp6(exact)}, в программе ${N(z)}`);
        return z;
      });
      ['z_i', 'z_f', 'z_c', 'z_o'].forEach((name, k) => { P[name] = cell.z[k]; });
      GATES.forEach(([key, , fn], k) => { cell.vals[key] = ` = ${fn}(${N(cell.z[k])})`; });
      const rows = GATES.map(([, sym], k) => {
        const n = col(sym), exact = x[0] * W[0][k] + x[1] * W[1][k] + cell.hp * U[k] + b[k], tail = N10(exact) === N(cell.z[k]) ? '' : ` = ${N10(exact)}`;
        return `z_${n} = x_t W_${n} + h_{t-1} U_${n} + b_${n} &= ${M(x[0])}\\cdot ${M(W[0][k])} + ${M(x[1])}\\cdot ${M(W[1][k])} + ${M(cell.hp)}\\cdot ${M(U[k])} + ${N(b[k])}${tail} = ${N(cell.z[k])}`;
      });
      wordFull.push(aligned(rows));
      return lstm(['i', 'f', 'c_hat', 'o'], { sub: aligned(rows), cap: 'Аргумент каждого вентиля z: x_t W_i — число за числом вектора слова на столбец i таблицы W (x_{t,0}·W_{0,i} + x_{t,1}·W_{1,i}), плюс h_{t−1} на столбец i таблицы U, плюс смещение. Порядок как в коде и в Keras. Если точная сумма длиннее 6 цифр, она показана перед округлением. Программа печатает z как z_i, z_f, z_c, z_o.' });
    }
    if (at('print(f"   z_i')) return printed(lstmTitle(), system(LSTM, cell.vals, ['i', 'f', 'c_hat', 'o']));
    const gate = !inFn || inFn === 'lstm_step' ? GATES.find(([key]) => at(key + ' = r(')) : null;
    if (gate) {
      const [key, sym, fn] = gate, k = GATES.indexOf(gate), z = cell.z[k];
      let sub, val;
      if (fn === '\\sigma') { const e = Math.exp(-z); val = check(key, v[key], 1 / (1 + e), 1 / (1 + R10(e))); sub = `${sym}_t = ${sigma(z, e, val)}`; }
      else {
        const a = Math.exp(z), b = Math.exp(-z); val = check(key, v[key], Math.tanh(z), (R10(a) - R10(b)) / (R10(a) + R10(b)));
        sub = `\\tilde c_t = \\tanh(${N(z)}) = \\frac{e^{${N(z)}} - e^{${N(0 - z)}}}{e^{${N(z)}} + e^{${N(0 - z)}}} = \\frac{${N10(a)} - ${N10(b)}}{${N10(a)} + ${N10(b)}} = ${N(val)}`;
      }
      cell[key] = val; cell.vals[key] += ` = ${N(val)}`; P[key === 'c_hat' ? 'c~' : key] = val; gateRows.push(sub.replace(' = ', ' &= '));
      const caps = { i: 'Входной вентиль: какую долю нового записать в память (0…1).', f: 'Вентиль забывания: какую долю старой памяти оставить (0…1).',
        c_hat: 'Кандидат в память: что записать (−1…1). В коде c_hat, в выводе c~, в формулах c̃_t — одно и то же.', o: 'Выходной вентиль: какую долю памяти показать наружу (0…1).' };
      return lstm([key], { sub, cap: caps[key], w: [fn === '\\sigma' ? 'sigmoid' : 'tanh-activation'] });
    }
    if (at('print(f"   i = ')) return printed(lstmTitle(), system(LSTM, cell.vals, ['i', 'f', 'c_hat', 'o']));
    if (at('c = r(f')) {
      const a = cell.f * cell.cp, b = cell.i * cell.c_hat;
      cell.c = check('c_t', v.c, cell.f * cell.cp + cell.i * cell.c_hat, R10(a) + R10(b)); cell.vals.c = ` = ${N(cell.c)}`; P.c = cell.c;
      const sub = `c_t = ${N(cell.f)}\\cdot ${M(cell.cp)} + ${N(cell.i)}\\cdot ${M(cell.c_hat)} = ${M10(a)} + ${M10(b)} = ${N(cell.c)}`;
      wordFull.push(aligned(gateRows), sub);
      return lstm(['c'], { sub, cap: 'Новая память: старая, умноженная на f, плюс кандидат, умноженный на i. ⊙ — поэлементно; в книге векторы из 32 чисел, здесь по одному.' });
    }
    if (at('h = r(o')) {
      const th = Math.tanh(cell.c); cell.h = check('h_t', v.h, cell.o * th, cell.o * R10(th)); cell.vals.h = ` = ${N(cell.h)}`; P.h = cell.h;
      const sub = `h_t = ${N(cell.o)}\\cdot\\tanh(${N(cell.c)}) = ${N(cell.o)}\\cdot ${M10(th)} = ${N(cell.h)}`;
      wordFull.push(sub);
      return lstm(['h'], { sub, cap: 'Выход ячейки: сжатая tanh память, пропущенная через выходной вентиль.' });
    }
    if (at('print(f"   c = ')) return printed(lstmTitle(), system(LSTM, cell.vals, ['c', 'h']));
    if (at('return h, c, [i')) return lstm([], { sub: `(h_t,\\ c_t) = (${N(cell.h)},\\ ${N(cell.c)})`, cap: 'Шаг t закончен: функция отдаёт выход, память и вентили (их запомнит обратный проход).' });
    if (at('h_new, c_new, gates = lstm_step')) { S.h = check('h', v.h_new, cell.h); S.c = check('c', v.c_new, cell.c); return lstm([], { sub: `h_${t} = ${N(S.h)},\\quad c_${t} = ${N(S.c)}`, cap: 'Результат слова: выход и память. Для следующего слова это будут h_{t−1} и c_{t−1}.' }); }
    if (at('steps.append(')) return lstm([], { cap: 'Запомнили для обратного прохода: номер слова, h_{t−1}, c_{t−1}, i, f, c̃, o, c_t, h_t.' });
    if (at('h, c = h_new, c_new')) {
      s.k = { title: lstmTitle(), full: wordFull, w: ['sigmoid', 'tanh-activation'],
        cap: `Слово «${cell.word}» целиком: вход, аргументы вентилей, вентили, память, выход. Каждое число считается из показанных выше и напечатано программой; h_t и c_t пойдут в следующее слово.` };
      return lstm([], { sub: `h_${t} = ${N(S.h)},\\quad c_${t} = ${N(S.c)}`, cap: 'Для следующего слова это будут h_{t−1} и c_{t−1}.' });
    }
    if (at('H.append')) { S.H.push(S.h); return { title: `${epochTitle()} → выход LSTM`, rows: [{ t: `h^{(${review})} = h_T = ${N(S.h)}`, s: 'cur' }], sub: hRow(), cap: 'По одному числу на отзыв — всё, что увидит классификатор. В коде это список H; H[k] в формулах — h^{(k)}.' }; }
    if (at('mem.append')) return { title: `${epochTitle()} → выход LSTM`, rows: [{ t: `h^{(${review})} = h_T = ${N(S.h)}`, s: 'cur' }], cap: 'Записи всех слов отзыва ушли в mem: обратный проход пойдёт по ним в обратном порядке.' };
    if (at('print(f"   h_T')) {
      P.h_T = S.h;
      const m = { title: `${epochTitle()} → выход LSTM`, rows: [{ t: `h^{(${review})} = h_T = ${N(S.h)}`, s: 'cur' }], sub: hRow(), cap: 'По одному числу на отзыв — всё, что увидит классификатор. В коде это список H; H[k] в формулах — h^{(k)}.' };
      s.k = { title: `${epochTitle()} → выход LSTM · отзыв ${review} «${sentence}»`, cap: `Слова кончились: классификатор получит только последний выход h_T = ${N(S.h)}. ${m.cap}` };
      return m;
    }

    // ── Dense, ошибка, градиент Dense ──
    if (at('p = [r(')) {
      const sub = [];
      S.p = S.H.map((h, k) => {
        const z = S.w * h + S.b, eShown = Math.exp(-R10(z));
        const val = check(`p_${k + 1}`, v.p[k], 1 / (1 + Math.exp(-z)), 1 / (1 + R10(eShown)));
        P[`p_${k + 1}`] = val;
        sub.push(`p_${k + 1} &= \\sigma(${M(S.w)}\\cdot ${M(h)} + ${M(S.b)}) = \\sigma(${N10(z)})`, `&= ${sigma(z, eShown, val, N10).replace(/^\\sigma\([^)]*\) = /, '')}`);
        return val;
      });
      train.p = ` = (${S.p.map(N).join(',\\ ')})`;
      return dn(dense(['p'], { sub: aligned(sub), cap: 'Прямой проход до конца: Dense + sigmoid дают вероятность того, что отзыв позитивный. В коде p — список p_1, p_2.', w: ['sigmoid'] }));
    }
    if (at('print(f"   p_1')) return printed(`${epochTitle()} → Dense и ошибка`, system(DENSE, train, ['p']));
    if (at('acc = ')) {
      const y = g.labels, n = S.p.length, guess = S.p.map(p => p > 0.5 ? 1 : 0), hits = guess.filter((q, k) => q === y[k]).length;
      const val = check('acc', v.acc, hits / n); train.acc = ` = ${N(val)}`; P.acc = val;
      const sub = S.p.map((p, k) => `p_${k + 1} = ${N(p)} ${guess[k] ? '>' : '\\le'} 0{.}5 &\\Rightarrow ${guess[k]}\\ ${guess[k] === y[k] ? '=' : '\\ne'}\\ y_${k + 1} = ${y[k]}\\ ${guess[k] === y[k] ? '\\checkmark' : '\\times'}`);
      sub.push(`\\mathrm{acc} &= \\tfrac{${hits}}{${n}} = ${N(val)}`);
      return dn(dense(['acc'], { sub: aligned(sub), cap: 'metrics=["acc"]: доля отзывов, угаданных при пороге 0.5 (ровно 0.5 — класс 0); [условие] в формуле — 1, если верно, иначе 0. На обучение не влияет — Keras её только печатает.' }));
    }
    if (at('loss = r(bce')) {
      const y = g.labels, n = S.p.length;
      const terms = S.p.map((p, k) => y[k] * Math.log(p) + (1 - y[k]) * Math.log(1 - p));
      const logs = S.p.map((p, k) => y[k] ? Math.log(p) : Math.log(1 - p));
      S.L = check('L', v.loss, -sum(terms) / n, -sum(logs.map(R10)) / n); train.L = ` = ${N(S.L)}`; P.loss = S.L;
      const full = S.p.map((p, k) => `${y[k]}\\cdot\\ln ${N(p)} + ${1 - y[k]}\\cdot\\ln(1-${N(p)})`).join(' + ');
      const kept = S.p.map((p, k) => y[k] ? `\\ln ${N(p)}` : `\\ln ${N(1 - p)}`);
      return dn(dense(['L'], { w: ['logistic-cost'],
        sub: aligned([`L &= -\\tfrac1${n}\\big[${full}\\big]`, `&= -\\tfrac1${n}\\big[${kept.join(' + ')}\\big] = -\\tfrac1${n}\\big[${logs.map(M10).join(' + ')}\\big] = ${N(S.L)}`]),
        cap: `ln — натуральный логарифм. y = 1 оставляет ln p, y = 0 — ln(1 − p). N = ${n}: оба отзыва в одном батче. В коде это loss — то число, которое Keras печатает как loss эпохи; оно посчитано до шага обучения.` }));
    }
    if (at('print(f"   acc')) return printed(`${epochTitle()} → Dense и ошибка`, system(DENSE, train, ['acc', 'L']));
    if (at('dz_d = [r(')) {
      const y = g.labels, n = S.p.length;
      S.dz = S.p.map((p, k) => check(`\\delta z^d_${k + 1}`, v.dz_d[k], (p - y[k]) / n)); P.dz_d = S.dz; train.dz_d = ` = (${S.dz.map(N).join(',\\ ')})`;
      return dn(dense(['dz_d'], { w: ['logistic-gradient-component'], sub: aligned(S.p.map((p, k) => `\\delta z^d_${k + 1} &= \\frac{${N(p)} - ${y[k]}}{${n}} = ${N(S.dz[k])}`)),
        cap: 'Начало обратного прохода. Для пары sigmoid + кросс-энтропия производная loss по аргументу сигмоиды отзыва k равна (p_k − y_k)/N (вывод — по ссылке ниже). Дальше каждый градиент — по правилу цепочки от этого числа.' }));
    }
    if (at('g_wd = r(sum')) {
      const parts = S.dz.map((d, k) => d * S.H[k]);
      S.g_wd = check('g_{w_d}', v.g_wd, sum(parts), sum(parts.map(R10))); P.g_wd = S.g_wd; train.g_wd = ` = ${N(S.g_wd)}`;
      return dn(dense(['g_wd'], { sub: `g_{w_d} = ${S.dz.map((d, k) => `${M(d)}\\cdot ${M(S.H[k])}`).join(' + ')} = ${parts.map(M10).join(' + ')} = ${N(S.g_wd)}`, cap: 'Градиент по весу Dense: δz^d умножить на вход h_T, сумма по отзывам.' }));
    }
    if (at('g_bd = r(sum')) {
      S.g_bd = check('g_{b_d}', v.g_bd, sum(S.dz)); P.g_bd = S.g_bd; train.g_bd = ` = ${N(S.g_bd)}`;
      return dn(dense(['g_bd'], { sub: `g_{b_d} = ${S.dz.map(M).join(' + ')} = ${N(S.g_bd)}`, cap: 'Градиент по смещению Dense: его «вход» равен 1.' }));
    }
    if (at('print(f"   dz_d')) {
      s.k = { title: `${epochTitle()} → Dense, ошибка и её градиент`, full: denseFull, w: ['sigmoid', 'logistic-cost', 'logistic-gradient-component'],
        cap: 'Прямой проход до конца, метрика, ошибка — и первый шаг назад: δz^d и градиенты Dense. Строку «acc … loss» Keras печатает в конце эпохи; они посчитаны до шага rmsprop.' };
      return printed(`${epochTitle()} → Dense и ошибка`, system(DENSE, train, ['dz_d', 'g_wd', 'g_bd']));
    }
    if (at('gE = [[0.0') || at('gW = [[0.0') || at('gU = [0.0') || at('gb = [0.0')) return { title: `${epochTitle()} → назад`, rows: system(BACK, {}, []), cap: 'Градиенты весов ячейки и таблицы E: сначала нули, вклады каждого слова будут прибавляться.' };

    // ── обратный проход ──
    if (at('for k, steps in enumerate(mem)')) {
      if (s.x) return { title: `${epochTitle()} → назад`, rows: system(BACK, {}, []), cap: 'Обратный проход по обоим отзывам закончен: градиенты gE, gW, gU, gb собраны (пока не округлены).' };
      review = v.k + 1; sentence = g.sentences[v.k]; back = {}; bvals = {};
      return { title: backTitle(), rows: system(BACK, {}, []), cap: 'Обратный проход по отзыву: от последнего слова к первому.' };
    }
    if (at('dh, dc = r(dz_d[k] * w_d)')) {
      S.dh = check('\\delta h_T', v.dh, S.dz[v.k] * S.w); S.dc = 0; P.dh = S.dh; P.dc = 0; train.dh = ` = ${N(S.dh)}`;
      return dense(['dh'], { sub: `\\delta h^{(${review})}_T = ${M(S.dz[v.k])}\\cdot ${M(S.w)} = ${N(S.dh)},\\quad \\delta c_T = 0`, cap: 'loss зависит от последнего выхода h_T только через Dense: δh_T = δz^d·w_d. Память c_T дальше никуда не идёт, её δ = 0.' });
    }
    if (at('print(f" отзыв {k + 1}: назад')) return { title: backTitle(), rows: system(BACK, {}, []), cap: 'Заголовок обратного прохода по отзыву и два числа, с которых он начинается.' };
    if (at('for idx, hp, cp, i, f, c_hat, o, c, h in reversed(steps)')) {
      if (s.x) return { title: backTitle(), rows: system(BACK, bvals, []), cap: 'Слова отзыва кончились: δh_0 и δc_0 дальше не нужны — перед первым словом памяти не было.' };
      back = { t: v.steps.length - v.steps.findIndex(st => st[0] === v.idx && st[1] === v.hp && st[2] === v.cp), word: g.words[v.idx], idx: v.idx, x: g.E[v.idx], hp: v.hp, cp: v.cp, i: v.i, f: v.f, c_hat: v.c_hat, o: v.o, c: v.c, dh: S.dh, dc: S.dc };
      bvals = {};
      return bsys([], { sub: `t = ${back.t}:\\ \\delta h_t = ${N(back.dh)},\\quad \\delta c_t^{\\leftarrow} = ${N(back.dc)};\\quad i_t = ${N(back.i)},\\ f_t = ${N(back.f)},\\ \\tilde c_t = ${N(back.c_hat)},\\ o_t = ${N(back.o)},\\ c_t = ${N(back.c)},\\ c_{t-1} = ${N(back.cp)},\\ h_{t-1} = ${N(back.hp)}`,
        cap: `Слово «${back.word}» (t = ${back.t}) в обратном порядке. Из mem взяты его числа прямого прохода; δh_t и δc_t^{←} пришли справа — от Dense или от следующего слова.` });
    }
    if (at('print(f"{words[idx]:>5}: назад')) return bsys([], { cap: 'Заголовок слова в обратном проходе.' });
    if (at('def lstm_back')) {
      ['hp', 'cp', 'i', 'f', 'c_hat', 'o', 'c', 'dh', 'dc'].forEach(key => check(key, v[key], back[key]));
      backFull = [aligned([`t = ${back.t}\\ (\\text{${back.word}}):\\ \\delta h_t &= ${N(back.dh)},\\quad \\delta c_t^{\\leftarrow} = ${N(back.dc)}`,
        `x_t &= ${pair(back.x)},\\ h_{t-1} = ${N(back.hp)},\\ c_{t-1} = ${N(back.cp)},\\ i_t = ${N(back.i)},\\ f_t = ${N(back.f)},\\ \\tilde c_t = ${N(back.c_hat)},\\ o_t = ${N(back.o)},\\ c_t = ${N(back.c)}`])];
      return bsys([], { cap: 'Вызов функции: числа слова и пришедшие справа δh, δc стали её параметрами.' });
    }
    if (at('x = E[idx]') && inFn === 'lstm_back') return bsys([], { sub: `x_t = E[\\text{${back.word}}] = ${pair(back.x)}`, cap: 'Вектор слова понадобится для вкладов в g_W и для δx.' });
    if (at('tc = math.tanh(c)')) { back.tc = Math.tanh(back.c); return bsys(['do', 'dc'], { sub: `\\tanh(c_t) = \\tanh(${M(back.c)}) = ${N10(back.tc)}`, cap: 'tanh памяти нужен дважды: в δo и в δc. Программа его не округляет — показан с 10 цифрами.' }); }
    if (at('do = r(dh * tc)')) {
      back.do = check('\\delta o_t', v.do, back.dh * back.tc, back.dh * R10(back.tc)); P.do = back.do; bvals.do = ` = ${N(back.do)}`;
      return bk(bsys(['do'], { sub: `\\delta o_t = ${M(back.dh)}\\cdot ${M10(back.tc)} = ${N(back.do)}`, cap: 'h_t = o_t·tanh(c_t): производная по o_t — это δh_t, умноженная на tanh(c_t).' }));
    }
    if (at('dc = r(dc + dh * o')) {
      const d1 = 1 - back.tc * back.tc;
      back.dcn = check('\\delta c_t', v.dc, back.dc + back.dh * back.o * d1, back.dc + back.dh * back.o * R10(d1)); P.dc = back.dcn; bvals.dc = ` = ${N(back.dcn)}`;
      return bk(bsys(['dc'], { sub: `\\delta c_t = ${M(back.dc)} + ${M(back.dh)}\\cdot ${N(back.o)}\\cdot(1 - ${M10(back.tc)}^2) = ${M(back.dc)} + ${M(back.dh)}\\cdot ${N(back.o)}\\cdot ${N10(d1)} = ${N(back.dcn)}`,
        cap: 'Производная по памяти: что пришло справа (δc^{←}, от следующего слова через f_{t+1}), плюс от h_t через o_t·tanh′(c_t), а tanh′ = 1 − tanh².' }));
    }
    if (at('print(f"   do = ')) return printed(backTitle(), system(BACK, bvals, ['do', 'dc']));
    if (at('di = r(dc * c_hat)')) { back.di = check('\\delta i_t', v.di, back.dcn * back.c_hat); P.di = back.di; bvals.di = ` = ${N(back.di)}`; return bk(bsys(['di'], { sub: `\\delta i_t = ${M(back.dcn)}\\cdot ${M(back.c_hat)} = ${N(back.di)}`, cap: 'c_t = f·c_{t−1} + i·c̃: производная по i — δc_t·c̃_t.' })); }
    if (at('df = r(dc * cp)')) { back.df = check('\\delta f_t', v.df, back.dcn * back.cp); P.df = back.df; bvals.df = ` = ${N(back.df)}`; return bk(bsys(['df'], { sub: `\\delta f_t = ${M(back.dcn)}\\cdot ${M(back.cp)} = ${N(back.df)}`, cap: 'По f — δc_t·c_{t−1}. У первого слова c_0 = 0, поэтому там δf = 0.' })); }
    if (at('dc_hat = r(dc * i)')) { back.dch = check('\\delta\\tilde c_t', v.dc_hat, back.dcn * back.i); P['dc~'] = back.dch; bvals.dc_hat = ` = ${N(back.dch)}`; return bk(bsys(['dc_hat'], { sub: `\\delta \\tilde c_t = ${M(back.dcn)}\\cdot ${N(back.i)} = ${N(back.dch)}`, cap: 'По кандидату c̃ — δc_t·i_t.' })); }
    if (at('print(f"   di = ')) return printed(backTitle(), system(BACK, bvals, ['di', 'df', 'dc_hat']));
    if (at('dz = [r(di')) {
      const { i, f, c_hat, o } = back;
      const exact = [back.di * i * (1 - i), back.df * f * (1 - f), back.dch * (1 - c_hat * c_hat), back.do * o * (1 - o)];
      back.dz = exact.map((e, k) => check(`\\delta z_${GK[k]}`, v.dz[k], e));
      ['dz_i', 'dz_f', 'dz_c', 'dz_o'].forEach((name, k) => { P[name] = back.dz[k]; bvals[name] = ` = ${N(back.dz[k])}`; });
      const rows = [
        `\\delta z_i &= ${M(back.di)}\\cdot ${N(i)}\\cdot(1 - ${N(i)}) = ${M(back.di)}\\cdot ${N10(i * (1 - i))} = ${N(back.dz[0])}`,
        `\\delta z_f &= ${M(back.df)}\\cdot ${N(f)}\\cdot(1 - ${N(f)}) = ${M(back.df)}\\cdot ${N10(f * (1 - f))} = ${N(back.dz[1])}`,
        `\\delta z_c &= ${M(back.dch)}\\cdot(1 - ${M(c_hat)}^2) = ${M(back.dch)}\\cdot ${N10(1 - c_hat * c_hat)} = ${N(back.dz[2])}`,
        `\\delta z_o &= ${M(back.do)}\\cdot ${N(o)}\\cdot(1 - ${N(o)}) = ${M(back.do)}\\cdot ${N10(o * (1 - o))} = ${N(back.dz[3])}`];
      // Путь калькулятора: через показанные 10-значные множители.
      [back.di * R10(i * (1 - i)), back.df * R10(f * (1 - f)), back.dch * R10(1 - c_hat * c_hat), back.do * R10(o * (1 - o))].forEach((sh, k) => check(`\\delta z_${GK[k]} (через показанные)`, back.dz[k], sh));
      return bk(bsys(['dz_i', 'dz_f', 'dz_c', 'dz_o'], { sub: aligned(rows), w: ['sigmoid-derivative'], cap: 'Назад через активации к аргументам вентилей: σ′(z) = σ(1 − σ), tanh′(z) = 1 − tanh². Значения σ и tanh — это сами i, f, o и c̃, они уже посчитаны.' }));
    }
    if (at('print(f"   dz_i')) return printed(backTitle(), system(BACK, bvals, ['dz_i', 'dz_f', 'dz_c', 'dz_o']));
    if (at('for k in range(4):') && inFn === 'lstm_back') {
      if (s.x) return bsys(['gW', 'gU', 'gb'], { cap: 'Вклады этого слова во все четыре столбца прибавлены.' });
      return bsys(['gW', 'gU', 'gb'], { cap: `Столбец k = ${GK[v.k]}: вклад этого слова в градиенты весов ячейки.` });
    }
    if (at('gW[0][k] += ')) { const k = v.k; return bsys(['gW'], { sub: `g_{W_{0,${GK[k]}}} \\mathrel{+}= x_{t,0}\\,\\delta z_${GK[k]} = ${M(back.x[0])}\\cdot ${M(back.dz[k])} = ${M10(back.x[0] * back.dz[k])}`, cap: 'z = x·W + …: производная по W_{0,k} — это x_{t,0}·δz_k. Вклад прибавляется к сумме по всем словам; итог — в шаге «градиенты всех весов».' }); }
    if (at('gW[1][k] += ')) { const k = v.k; return bsys(['gW'], { sub: `g_{W_{1,${GK[k]}}} \\mathrel{+}= x_{t,1}\\,\\delta z_${GK[k]} = ${M(back.x[1])}\\cdot ${M(back.dz[k])} = ${M10(back.x[1] * back.dz[k])}`, cap: 'То же для второго числа вектора слова.' }); }
    if (at('gU[k] += ')) { const k = v.k; return bsys(['gU'], { sub: `g_{U_${GK[k]}} \\mathrel{+}= h_{t-1}\\,\\delta z_${GK[k]} = ${M(back.hp)}\\cdot ${M(back.dz[k])} = ${M10(back.hp * back.dz[k])}`, cap: 'Производная по U_k — h_{t−1}·δz_k.' }); }
    if (at('gb[k] += ')) { const k = v.k; return bsys(['gb'], { sub: `g_{b_${GK[k]}} \\mathrel{+}= \\delta z_${GK[k]} = ${M(back.dz[k])}`, cap: 'Производная по смещению — сам δz_k.' }); }
    if (at('dx = [r(sum(W[j][k]')) {
      const { W } = g;
      back.dx = [0, 1].map(j => { const terms = [0, 1, 2, 3].map(k => W[j][k] * back.dz[k]); return check(`\\delta x_{t,${j}}`, v.dx[j], sum(terms), sum(terms.map(R10))); });
      P.dx = back.dx; bvals.dx = ` = ${pair(back.dx)}`;
      // Две строки на число: произведения, затем их значения и сумма — иначе строка не влезает в панель.
      const rows = [0, 1].flatMap(j => [`\\delta x_{t,${j}} &= ${[0, 1, 2, 3].map(k => `${M(W[j][k])}\\cdot ${M(back.dz[k])}`).join(' + ')}`, `&= ${[0, 1, 2, 3].map(k => M10(W[j][k] * back.dz[k])).join(' + ')} = ${N(back.dx[j])}`]);
      backFull.push(aligned([
        ...[0, 1].flatMap(j => [`g_{W_{${j},k}} &\\mathrel{+}= x_{t,${j}}\\,\\delta z_k = ${M(back.x[j])}\\cdot(${back.dz.map(M).join(',\\ ')})`, `&= (${back.dz.map(d => M10(back.x[j] * d)).join(',\\ ')})`]),
        `g_{U_k} &\\mathrel{+}= h_{t-1}\\,\\delta z_k = ${M(back.hp)}\\cdot(${back.dz.map(M).join(',\\ ')})`, `&= (${back.dz.map(d => M10(back.hp * d)).join(',\\ ')})`,
        `g_{b_k} &\\mathrel{+}= \\delta z_k = (${back.dz.map(M).join(',\\ ')})`]));
      return bk(bsys(['dx'], { sub: aligned(rows), cap: `Градиент по вектору слова: δx_j = Σ_k W_{jk}·δz_k — строка j таблицы W на вектор δz. Он прибавится к строке «${back.word}» таблицы E (g_E).` }));
    }
    if (at('gE[idx][0] += ') || at('gE[idx][1] += ')) { const j = at('gE[idx][0]') ? 0 : 1; return bsys(['dx'], { sub: `g_{E_{\\text{${back.word}},${j}}} \\mathrel{+}= \\delta x_{t,${j}} = ${M(back.dx[j])}`, cap: 'Строка Embedding этого слова: вклад прибавляется (слово «movie» встречается в обоих отзывах — у него два вклада).' }); }
    if (at('dh_prev = r(sum(U[k]')) {
      const { U } = g, terms = [0, 1, 2, 3].map(k => U[k] * back.dz[k]);
      back.dh_prev = check('\\delta h_{t-1}', v.dh_prev, sum(terms), sum(terms.map(R10))); P.dh_prev = back.dh_prev; bvals.dh_prev = ` = ${N(back.dh_prev)}`;
      const sub = aligned([`\\delta h_{t-1} &= ${[0, 1, 2, 3].map(k => `${M(U[k])}\\cdot ${M(back.dz[k])}`).join(' + ')}`, `&= ${terms.map(M10).join(' + ')} = ${N(back.dh_prev)}`]);
      return bk(bsys(['dh_prev'], { sub, cap: 'Что передать слову слева по выходу: z зависело от h_{t−1} через U, значит δh_{t−1} = Σ_k U_k·δz_k.' }));
    }
    if (at('dc_prev = r(dc * f)')) {
      back.dc_prev = check('\\delta c_{t-1}', v.dc_prev, back.dcn * back.f); P.dc_prev = back.dc_prev; bvals.dc_prev = ` = ${N(back.dc_prev)}`;
      const sub = `\\delta c_{t-1} = ${M(back.dcn)}\\cdot ${N(back.f)} = ${N(back.dc_prev)}`;
      return bk(bsys(['dc_prev'], { sub, cap: 'И по памяти: c_t = f_t·c_{t−1} + …, значит δc_{t−1} = δc_t·f_t. Это будет δc^{←} слова слева.' }));
    }
    if (at('print(f"   dx = ')) return printed(backTitle(), system(BACK, bvals, ['dx', 'dh_prev', 'dc_prev']));
    if (at('return dh_prev, dc_prev')) return bsys([], { sub: `(\\delta h_{t-1},\\ \\delta c_{t-1}) = (${N(back.dh_prev)},\\ ${N(back.dc_prev)})`, cap: 'Слово разобрано назад: функция отдаёт δ для слова слева.' });
    if (at('dh, dc = lstm_back(')) {
      S.dh = check('\\delta h', v.dh, back.dh_prev); S.dc = check('\\delta c', v.dc, back.dc_prev);
      backSteps.push({ word: back.word, idx: back.idx, x: back.x, hp: back.hp, dz: back.dz, dx: back.dx, review, t: back.t });
      s.k = { title: backTitle(), full: backFull, w: ['sigmoid-derivative', 'backpropagation'],
        cap: `Слово «${back.word}» назад целиком: от δh_t и δc_t к вентилям, к аргументам z, к вкладам в градиенты W, U, b, E — и к δh_{t−1}, δc_{t−1} для слова слева. Каждое число считается из показанных и напечатано программой.` };
      return bsys([], { sub: `\\delta h_{${back.t - 1}} = ${N(S.dh)},\\quad \\delta c_{${back.t - 1}} = ${N(S.dc)}`, cap: 'Для слова слева это будут δh_t и δc_t^{←}.' });
    }

    // ── градиенты всех весов ──
    if (at('gE = [[r(')) {
      const rows = [], tot = g.E.map(() => [0, 0]);
      backSteps.forEach(st => { tot[st.idx][0] += st.dx[0]; tot[st.idx][1] += st.dx[1]; });
      S.gE = tot.map((row, w) => row.map((x, j) => check(`g_{E_{${g.words[w]},${j}}}`, v.gE[w][j], x))); P.gE = S.gE;
      g.words.forEach((w, a) => [0, 1].forEach(j => {
        const terms = backSteps.filter(st => st.idx === a).map(st => M(st.dx[j]));
        rows.push(`g_{E_{\\text{${w}},${j}}} &= ${terms.join(' + ')} = ${N(S.gE[a][j])}`);
      }));
      S.gradRows = [aligned(rows)];
      return { title: `${epochTitle()} → градиенты всех весов`, rows: system(BACK, {}, ['dx']), sub: aligned(rows), cap: 'Градиент таблицы E: по строке слова — сумма δx всех его появлений (в порядке обратного прохода). Здесь суммы округляются до 6 цифр.' };
    }
    if (at('gW = [[r(')) {
      const rows = [];
      S.gW = [0, 1].map(j => [0, 1, 2, 3].map(k => {
        const terms = backSteps.map(st => st.x[j] * st.dz[k]);
        const val = check(`g_{W_{${j},${GK[k]}}}`, v.gW[j][k], sum(terms), sum(terms.map(R10)));
        rows.push(`g_{W_{${j},${GK[k]}}} &= ${backSteps.map(st => `${M(st.x[j])}\\cdot ${M(st.dz[k])}`).join(' + ')}`, `&= ${terms.map(M10).join(' + ')} = ${N(val)}`);
        return val;
      })); P.gW = S.gW;
      S.gradRows.push(aligned(rows));
      return { title: `${epochTitle()} → градиенты всех весов`, rows: system(BACK, {}, ['gW']), sub: aligned(rows), cap: 'Градиент W: для каждой ячейки таблицы — сумма вкладов x_{t,j}·δz_k по всем шести словам обоих отзывов (в порядке обратного прохода: отзыв 1 с конца, затем отзыв 2 с конца).' };
    }
    if (at('gU = [r(')) {
      const rows = [];
      S.gU = [0, 1, 2, 3].map(k => {
        const terms = backSteps.map(st => st.hp * st.dz[k]);
        const val = check(`g_{U_${GK[k]}}`, v.gU[k], sum(terms), sum(terms.map(R10)));
        rows.push(`g_{U_${GK[k]}} &= ${backSteps.map(st => `${M(st.hp)}\\cdot ${M(st.dz[k])}`).join(' + ')}`, `&= ${terms.map(M10).join(' + ')} = ${N(val)}`);
        return val;
      }); P.gU = S.gU;
      S.gradRows.push(aligned(rows));
      return { title: `${epochTitle()} → градиенты всех весов`, rows: system(BACK, {}, ['gU']), sub: aligned(rows), cap: 'Градиент U: сумма h_{t−1}·δz_k по словам. У первых слов h_0 = 0 — их вклад нулевой.' };
    }
    if (at('gb = [r(')) {
      const rows = [];
      S.gb = [0, 1, 2, 3].map(k => {
        const terms = backSteps.map(st => st.dz[k]);
        const val = check(`g_{b_${GK[k]}}`, v.gb[k], sum(terms));
        rows.push(`g_{b_${GK[k]}} &= ${terms.map(M).join(' + ')} = ${N(val)}`);
        return val;
      }); P.gb = S.gb;
      S.gradRows.push(aligned(rows));
      return { title: `${epochTitle()} → градиенты всех весов`, rows: system(BACK, {}, ['gb']), sub: aligned(rows), cap: 'Градиент смещений: просто сумма δz_k по словам.' };
    }
    if (at('print(f"   gE = ')) return printed(`${epochTitle()} → градиенты всех весов`, system(BACK, {}, ['dx']));
    if (at('print(f"   gW = ')) {
      s.k = { title: `${epochTitle()} → градиенты всех весов`, full: S.gradRows, w: ['backpropagation'],
        cap: 'Все градиенты эпохи: E (по строкам слов), W, U, b — суммы вкладов шести слов, и g_{w_d}, g_{b_d} из шага Dense. Дальше каждый вес сдвинется на свой градиент.' };
      return printed(`${epochTitle()} → градиенты всех весов`, system(BACK, {}, ['gW', 'gU', 'gb']));
    }

    // ── шаг rmsprop ──
    if (at('for w in range(max_words)')) {
      if (!upd) computeUpdate(Object.assign({}, g, { gE: S.gE, gW: S.gW, gU: S.gU, gb: S.gb, g_wd: S.g_wd, g_bd: S.g_bd, w_d: S.w, b_d: S.b }));
      if (s.x) return { title: `${epochTitle()} → шаг rmsprop`, rows: system(UPDATE, {}, []), cap: 'Таблица E обновлена.' };
      return { title: `${epochTitle()} → шаг rmsprop · E`, rows: system(UPDATE, {}, ['v', 'theta']), cap: `Строка «${g.words[v.w]}» таблицы E: два веса, у каждого свой v.` };
    }
    if (at('for j in range(2):')) return { title: `${epochTitle()} → шаг rmsprop`, rows: system(UPDATE, {}, ['v', 'theta']), cap: s.x ? 'Строка закончена.' : `j = ${v.j}.` };
    if (at('for k in range(4):')) return { title: `${epochTitle()} → шаг rmsprop`, rows: system(UPDATE, {}, ['v', 'theta']), cap: s.x ? 'Столбцы закончились.' : `Столбец k = ${GK[v.k]}.` };
    if (at('def rms')) { const { name, u } = updTarget(s.call); back.upd = { name, u }; return { title: `${epochTitle()} → шаг rmsprop · ${plain(name)}`, rows: system(UPDATE, {}, ['v', 'theta']), sub: `\\theta = ${name} = ${M(u.old)},\\quad v = ${N(u.vOld)},\\quad g = ${M(u.g)}`, cap: 'Вызов rms(): вес, его v и его градиент.' }; }
    if (at('v = r(rho * v')) {
      const { name, u } = back.upd; check(`v_{${name}}`, v.v, u.v);
      return { title: `${epochTitle()} → шаг rmsprop · ${plain(name)}`, rows: system(UPDATE, { v: ` = ${N(u.v)}` }, ['v']), sub: updSub(name, u).replace(/\\\\.*$/, '\\end{aligned}'), cap: 'ρ = 0.9: средний квадрат градиента помнит прошлое на 90 %.' };
    }
    if (at('return r(theta - lr')) {
      const { name, u } = back.upd, got = Number(String(s.r).replace(/^\(/, '').split(',')[0]);
      check(name, got, u.theta);
      return { title: `${epochTitle()} → шаг rmsprop · ${plain(name)}`, rows: system(UPDATE, { v: ` = ${N(u.v)}`, theta: ` = ${N(u.theta)}` }, ['theta']), sub: updSub(name, u), cap: 'η = 0.001, ε = 10⁻⁷ — как в Keras 3, ε под корнем. Градиент делится на корень из своего среднего квадрата, поэтому размер шага почти не зависит от величины g: в первой эпохе v = 0.1·g², шаг = η/√0.1 ≈ 0.00316 в сторону, противоположную градиенту.' };
    }
    if (/^(E\[w\]\[j\]|W\[j\]\[k\]|U\[k\]|b\[k\]|w_d, v_wd|b_d, v_bd)/.test(line) && line.includes('= rms(')) {
      const { name, u } = back.upd;
      return { title: `${epochTitle()} → шаг rmsprop · ${plain(name)}`, rows: system(UPDATE, { v: ` = ${N(u.v)}`, theta: ` = ${N(u.theta)}` }, []), sub: `${name} \\leftarrow ${N(u.theta)},\\quad v_{${name}} \\leftarrow ${N(u.v)}`, cap: 'Новый вес и новый v записаны на место старых.' };
    }
    if (at('print(f"   vE = ') || at('print(f"   vW = ') || at('print(f"   vU = ')) return printed(`${epochTitle()} → шаг rmsprop`, system(UPDATE, {}, []), 'Таблицы v и новые веса после шага — те же числа, что в формулах выше.');
    if (at('print(f"   v_wd = ')) { S.w = upd.w_d.theta; S.b = upd.b_d.theta; return printed(`${epochTitle()} → шаг rmsprop`, system(UPDATE, {}, []), 'Dense после шага. Все веса обновлены: следующая эпоха начнёт прямой проход уже с них.'); }
    if (at('print(f"1/1')) {
      s.k = { title: `${epochTitle()} → шаг rmsprop: все веса`, full: updTables(), cap: 'Один шаг rmsprop для каждого из 28 весов: v ← 0.9·v + 0.1·g², вес ← вес − 0.001·g/√(v + 10⁻⁷). Таблицы: градиент, новый v, новый вес. Строка «1/1 - acc … loss» — как у Keras: acc и loss посчитаны до шага.' };
      return { title: epochTitle(), rows: system(UPDATE, {}, []), cap: 'Та же строка, что печатает Keras: acc и loss этой эпохи, посчитанные до шага rmsprop.' };
    }
    if (at('print("после обучения:")')) return { title: 'После fit', rows: [], cap: 'Итоговые веса с 4 знаками — как печатает их сокращённый оригинал на Keras выше.' };
    if (at('print("E =", r4(E))')) { S.final = [`E = ${table(['j{=}0', 'j{=}1'], g.E.map((row, k) => [`\\text{${g.words[k]}}`, row]), N4)}`]; return { title: 'После fit', rows: [{ t: S.final[0], s: 'cur' }], cap: 'Таблица E после трёх эпох (4 знака) — сверьте с выводом Keras.' }; }
    if (at('print("W =", r4(W))')) { S.final.push(`W = ${table(GATE_COLS, [['x_{t,0}', g.W[0]], ['x_{t,1}', g.W[1]]], N4)}`); return { title: 'После fit', rows: [{ t: S.final[1], s: 'cur' }], cap: 'W после обучения.' }; }
    if (at('print("U =", r4(U))')) { S.final.push(`U = ${table(GATE_COLS, [['h_{t-1}', g.U]], N4)}`); return { title: 'После fit', rows: [{ t: S.final[2], s: 'cur' }], cap: 'U после обучения.' }; }
    if (at('print("b =", r4(b))')) { S.final.push(`b = ${table(GATE_COLS, [['1', g.b]], N4)}`); return { title: 'После fit', rows: [{ t: S.final[3], s: 'cur' }], cap: 'b после обучения.' }; }
    if (at('print("w_d =", r4(w_d))')) { S.final.push(`w_d = ${N4(g.w_d)}`); return { title: 'После fit', rows: [{ t: S.final[4], s: 'cur' }], cap: 'w_d после обучения.' }; }
    if (at('print("b_d =", r4(b_d))')) {
      S.final.push(`b_d = ${N4(g.b_d)}`);
      s.k = { title: 'После fit', rows: S.final.map(t => ({ t, s: 'cur' })), cap: 'Все веса после трёх эпох с 4 знаками — те же, что печатает Keras в «после обучения» (он считает в float32, поэтому сверка идёт по 4 знакам).' };
      return { title: 'После fit', rows: [{ t: S.final[5], s: 'cur' }], cap: 'b_d после обучения.' };
    }
    return null;
  }

  // v — переменные текущей функции, g — глобальные (внутри функций — с последнего шага модуля).
  // После каждого шага всё, что он напечатал как «имя = число», сверяется с числом страницы под этим именем.
  const lines = output.split('\n');
  const LIST = '\\[(?:[^\\[\\]]|\\[[^\\[\\]]*\\])*\\]';
  const PAIR = new RegExp(`(\\S+) = (${LIST}|-?\\d[\\d.e+-]*)`, 'g');
  let seen = 0;
  steps.forEach(s => {
    const v = s.n, g = s.f ? G : v;
    const line = src[s.l[0] - 1].trim();
    const m = math(line, s, v, g);
    if (m) s.m = m;
    if (!s.f) G = v;
    delete s.n;
    const four = line.includes('r4(');
    for (const out of lines.slice(seen, s.o)) for (const [, name, value] of out.matchAll(PAIR)) {
      if (!(name in P)) throw Error(`Печать «${name} = ${value}»: у страницы нет такого числа`);
      const want = pyList(P[name], four ? N4 : N);
      if (want !== value) throw Error(`Печать «${name} = ${value}»: на странице ${want}`);
    }
    seen = s.o;
  });
  if (!steps[steps.length - 1].k) throw Error('Последняя строка разбора не помечена шагом разбора (s.k)');
  return steps.filter(s => s.k).length;
};
