// Формулы сети к шагам мини-программы (только сборка, в браузер уходит готовый LaTeX).
// На каждом шаге: система уравнений текущей части сети, подсвечено уравнение этой строки,
// под ней — подстановка чисел. Числа с 6 знаками, и каждое показанное число дальше
// используется уже округлённым: пересчёт по экрану на калькуляторе даёт те же цифры.
// Итог каждой подстановки сверяется с программой до 4 знаков, иначе сборка падает.

const decimals = x => x === 0 ? 6 : Math.max(6, 5 - Math.floor(Math.log10(Math.abs(x))));
const R = x => { const v = Number(x.toFixed(decimals(x))); return v === 0 ? 0 : v; };
const N = x => { const v = R(x); return v.toFixed(decimals(v)).replace(/\.?0+$/, ''); };
const M = x => R(x) < 0 ? `(${N(x)})` : N(x);
const py4 = x => { const v = Number(x.toFixed(4)); return Number.isInteger(v) ? v.toFixed(1) : String(v === 0 ? 0 : v); };
const pair = v => `(${N(v[0])},\\ ${N(v[1])})`;

const GATES = [['i', 'i', '\\sigma'], ['f', 'f', '\\sigma'], ['c_hat', '\\tilde c', '\\tanh'], ['o', 'o', '\\sigma']];
const LSTM = [
  ['x', 'x_t = E[w_t]'],
  ...GATES.map(([key, sym, fn]) => [key, `${sym}_t = ${fn}(W_${sym === '\\tilde c' ? 'c' : sym} x_t + U_${sym === '\\tilde c' ? 'c' : sym} h_{t-1} + b_${sym === '\\tilde c' ? 'c' : sym})`]),
  ['c', 'c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde c_t'],
  ['h', 'h_t = o_t \\odot \\tanh(c_t)']
];
// Порядок как внутри model.fit: прямой проход, метрика, ошибка, градиент, шаг rmsprop.
const TRAIN = [
  ['p', 'p_k = \\sigma(w\\,h^{(k)} + b)'],
  ['acc', '\\mathrm{acc} = \\frac1N\\sum_k \\big[\\,[p_k > 0{.}5] = y_k\\,\\big]'],
  ['L', 'L = -\\frac1N\\sum_k \\big[y_k \\ln p_k + (1-y_k)\\ln(1-p_k)\\big]'],
  ['g_w', 'g_w = \\frac{\\partial L}{\\partial w} = \\frac1N\\sum_k (p_k - y_k)\\,h^{(k)}'],
  ['g_b', 'g_b = \\frac{\\partial L}{\\partial b} = \\frac1N\\sum_k (p_k - y_k)'],
  ['v_w', 'v_w \\leftarrow \\rho\\,v_w + (1-\\rho)\\,g_w^2'],
  ['v_b', 'v_b \\leftarrow \\rho\\,v_b + (1-\\rho)\\,g_b^2'],
  ['w', 'w \\leftarrow w - \\eta\\,\\frac{g_w}{\\sqrt{v_w + \\varepsilon}}'],
  ['b', 'b \\leftarrow b - \\eta\\,\\frac{g_b}{\\sqrt{v_b + \\varepsilon}}']
];
const table = (cols, rows) => `\\begin{array}{r|${'r'.repeat(cols.length)}} & ${cols.join(' & ')}\\\\ \\hline ${rows.map(([name, vals]) => `${name} & ${vals.map(N).join(' & ')}`).join('\\\\')}\\end{array}`;
const GATE_COLS = ['i', 'f', '\\tilde c', 'o'];
const sigma = (z, val) => `\\sigma(${N(z)}) = \\frac{1}{1+e^{${N(-z)}}} = \\frac{1}{1+${N(Math.exp(-z))}} = ${N(val)}`;
const aligned = rows => `\\begin{aligned}${rows.join('\\\\')}\\end{aligned}`;

module.exports = function addMath(steps, code) {
  const src = code.split('\n');
  let G = {}, review = 0, sentence = '', t = 0, epoch = 0, cell = {};
  const S = { H: [] };       // показанные значения: из них считаются следующие формулы
  let train = {};            // значения уже посчитанных строк текущей системы
  // Шаги разбора (s.k): данные, параметры, каждое слово, выход LSTM по отзыву, каждая эпоха, итог.
  // На странице такой шаг показывает подстановки всех своих строк разом (full) — слово или эпоху
  // можно пересчитать целиком, не листая по строке. Коллекторы копят подстановки до шага.
  let paramFull = [], wordFull = [], gateRows = [], epochFull = [];
  const param = m => { paramFull.push(m.sub); return m; };
  const ep = m => { epochFull.push(m.sub); return m; };
  const check = (what, actual, computed) => {
    if (py4(actual) !== py4(computed)) throw Error(`Формула ${what}: по показанным числам ${computed}, в программе ${actual}`);
    return R(computed);
  };
  const system = (rows, vals, cur) => rows.map(([key, tex]) => ({ t: tex + (vals[key] || ''), s: cur.includes(key) ? 'cur' : vals[key] ? 'done' : 'todo' }));
  const lstmTitle = () => `model.fit → LSTM · отзыв ${review} «${sentence}»` + (t ? ` · t = ${t}, слово «${cell.word}»` : '');
  const lstm = (cur, extra) => ({ title: lstmTitle(), rows: system(LSTM, cell.vals || {}, cur), ...extra });
  const epochTitle = () => `model.fit · эпоха ${epoch}`;

  function math(line, s, v, g) {
    const at = prefix => line.startsWith(prefix);
    if (at('print("train_pad:')) {
      s.k = { title: '1–3. Данные → номера', cap: 'Формул сети здесь нет: словарь, номер каждого слова и выравнивание нулями слева. Результат — в переменных: word2idx и train_pad.' };
      return null;
    }
    if (at('E = ')) return param({ title: 'Embedding: слово → вектор', rows: system([LSTM[0]], {}, ['x']),
      sub: `E = ${table(['x_{t,0}', 'x_{t,1}'], g.E.map((row, k) => [`\\text{${g.words[k]}}`, row]))}`,
      cap: 'Обучаемая таблица: строка на слово. В книге в строке 32 числа, здесь 2.' });
    if (at('W = ')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `W = ${table(GATE_COLS, [['x_{t,0}', g.W[0]], ['x_{t,1}', g.W[1]]])}`, cap: 'Веса от входа x_t: столбец на вентиль.' });
    if (at('U = ')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `U = ${table(GATE_COLS, [['h_{t-1}', g.U]])}`, cap: 'Веса от выхода прошлого шага h_{t−1}.' });
    if (at('b = ')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `b = ${table(GATE_COLS, [['1', g.b]])}`, cap: 'Смещения. У f смещение 1 — так Keras делает по умолчанию (unit_forget_bias).' });

    if (at('for seq')) {
      if (s.x) return null;
      review++; t = 0; cell = {}; sentence = g.sentences[review - 1];
      return lstm([], { cap: 'Новый отзыв: его слова по очереди пройдут через одну и ту же ячейку.' });
    }
    if (at('h, c = 0.0')) { S.h = 0; S.c = 0; return lstm([], { sub: 'h_0 = 0,\\quad c_0 = 0', cap: 'Перед первым словом памяти нет.' }); }
    if (at('for idx')) {
      if (s.x) return { title: `model.fit → выход LSTM · отзыв ${review}`, rows: [{ t: `h_T = h_${t} = ${N(S.h)}`, s: 'cur' }],
        cap: 'Слова кончились. Классификатор получит только последний выход h_T.' };
      t++; const idx = v.idx, x = g.E[idx];
      cell = { word: g.words[idx], x, vals: { x: ` = ${pair(x)}` } };
      return lstm(['x'], { sub: `x_${t} = E[\\text{${cell.word}}] = E_{${idx}} = ${pair(x)}`, cap: 'Embedding: номер слова выбирает строку таблицы E.' });
    }
    if (at('def lstm_step')) {
      cell.hp = check('h_{t-1}', v.h, S.h); cell.cp = check('c_{t-1}', v.c, S.c);
      wordFull = [`x_${t} = E[\\text{${cell.word}}] = ${pair(cell.x)},\\quad h_{${t - 1}} = ${N(cell.hp)},\\quad c_{${t - 1}} = ${N(cell.cp)}`]; gateRows = [];
      return lstm([], { sub: `x_t = ${pair(cell.x)},\\quad h_{t-1} = ${N(cell.hp)},\\quad c_{t-1} = ${N(cell.cp)}`, cap: 'Вход шага: вектор слова и память прошлого шага.' });
    }
    if (at('z = ')) {
      const { W, U, b } = g, x = cell.x;
      cell.z = GATES.map(([key], k) => check(`z_${key}`, v.z[k], x[0] * W[0][k] + x[1] * W[1][k] + cell.hp * U[k] + b[k]));
      GATES.forEach(([key, , fn], k) => { cell.vals[key] = ` = ${fn}(${N(cell.z[k])})`; });
      const rows = GATES.map(([, sym], k) => {
        const n = sym === '\\tilde c' ? 'c' : sym;
        return `W_${n} x_t + U_${n} h_{t-1} + b_${n} &= ${M(x[0])}\\cdot ${M(W[0][k])} + ${M(x[1])}\\cdot ${M(W[1][k])} + ${M(cell.hp)}\\cdot ${M(U[k])} + ${N(b[k])} = ${N(cell.z[k])}`;
      });
      wordFull.push(aligned(rows));
      return lstm(['i', 'f', 'c_hat', 'o'], { sub: aligned(rows), cap: 'Сначала аргумент каждого вентиля: входная часть + часть от прошлого h + смещение.' });
    }
    const gate = GATES.find(([key]) => at(key + ' = '));
    if (gate) {
      const [key, sym, fn] = gate, k = GATES.indexOf(gate), z = cell.z[k];
      let sub, val;
      if (fn === '\\sigma') { const e = R(Math.exp(-z)); val = check(key, v[key], 1 / (1 + e)); sub = `${sym}_t = ${sigma(z, val)}`; }
      else {
        const a = R(Math.exp(z)), b = R(Math.exp(-z)); val = check(key, v[key], (a - b) / (a + b));
        sub = `\\tilde c_t = \\tanh(${N(z)}) = \\frac{e^{${N(z)}} - e^{${N(-z)}}}{e^{${N(z)}} + e^{${N(-z)}}} = \\frac{${N(a)} - ${N(b)}}{${N(a)} + ${N(b)}} = ${N(val)}`;
      }
      cell[key] = val; cell.vals[key] += ` = ${N(val)}`; gateRows.push(sub.replace(' = ', ' &= '));
      const caps = { i: 'Входной вентиль: какую долю нового записать в память (0…1).', f: 'Вентиль забывания: какую долю старой памяти оставить (0…1).',
        c_hat: 'Кандидат в память: что записать (−1…1).', o: 'Выходной вентиль: какую долю памяти показать наружу (0…1).' };
      return lstm([key], { sub, cap: caps[key], w: [fn === '\\sigma' ? 'sigmoid' : 'tanh-activation'] });
    }
    if (at('c = f')) {
      const a = R(cell.f * cell.cp), b = R(cell.i * cell.c_hat);
      cell.c = check('c_t', v.c, a + b); cell.vals.c = ` = ${N(cell.c)}`;
      const sub = `c_t = ${N(cell.f)}\\cdot ${M(cell.cp)} + ${N(cell.i)}\\cdot ${M(cell.c_hat)} = ${N(a)} + ${M(b)} = ${N(cell.c)}`;
      wordFull.push(aligned(gateRows), sub);
      return lstm(['c'], { sub, cap: 'Новая память: старая, умноженная на f, плюс кандидат, умноженный на i. ⊙ — поэлементно; в книге векторы из 32 чисел, здесь по одному.' });
    }
    if (at('h = o')) {
      const th = R(Math.tanh(cell.c)); cell.h = check('h_t', v.h, cell.o * th); cell.vals.h = ` = ${N(cell.h)}`;
      const sub = `h_t = ${N(cell.o)}\\cdot\\tanh(${N(cell.c)}) = ${N(cell.o)}\\cdot ${M(th)} = ${N(cell.h)}`;
      wordFull.push(sub);
      return lstm(['h'], { sub, cap: 'Выход ячейки: сжатая tanh память, пропущенная через выходной вентиль.' });
    }
    if (at('return h, c')) return lstm([], { sub: `(h_t,\\ c_t) = (${N(cell.h)},\\ ${N(cell.c)})`, cap: 'Шаг t закончен: функция отдаёт выход и память.' });
    if (at('h, c = lstm_step')) {
      S.h = check('h', v.h, cell.h); S.c = check('c', v.c, cell.c);
      return lstm([], { sub: `h_${t} = ${N(S.h)},\\quad c_${t} = ${N(S.c)}`, cap: `Для следующего слова это будут h_{t−1} и c_{t−1}.` });
    }
    if (at('print(f"{words')) {
      s.k = { title: lstmTitle(), full: wordFull, w: ['sigmoid', 'tanh-activation'],
        cap: `Слово «${cell.word}» целиком: вход, аргументы вентилей, вентили, память, выход. Каждое число считается из показанных выше; h_t и c_t пойдут в следующее слово.` };
      return lstm([]);
    }
    if (at('H.append')) {
      S.H.push(S.h);
      const m = { title: 'model.fit → выход LSTM', rows: [{ t: `h^{(${review})} = h_T = ${N(S.h)}`, s: 'cur' }],
        sub: `H = (${S.H.map(N).join(',\\ ')})`, cap: 'По одному числу на отзыв — всё, что увидит классификатор.' };
      s.k = { title: `model.fit → выход LSTM · отзыв ${review} «${sentence}»`, cap: `Слова кончились: классификатор получит только последний выход h_T = ${N(S.h)}. ${m.cap}` };
      return m;
    }
    if (at('w_d, b_d = ')) { S.w = v.w_d; S.b = v.b_d; return param({ title: 'Dense: параметры', rows: system([TRAIN[0]], {}, []), sub: `w = ${N(S.w)},\\quad b = ${N(S.b)}`, cap: 'Веса слоя Dense(1): один вес на h и смещение. Считать по ним будем внутри fit.' }); }
    if (at('lr, rho, eps = ')) {
      S.lr = v.lr; S.rho = v.rho; S.eps = v.eps;
      const m = param({ title: 'compile: параметры rmsprop', rows: system(TRAIN.slice(5), {}, ['v_w', 'v_b', 'w', 'b']), sub: `\\eta = ${N(S.lr)},\\quad \\rho = ${N(S.rho)},\\quad \\varepsilon = 10^{-7}`,
        cap: 'optimizer="rmsprop" — это ровно эти числа, значения Keras по умолчанию. Сами формулы сработают в каждой эпохе fit.' });
      s.k = { title: '4–5. Модель и compile: все числа до обучения', rows: system(LSTM, {}, []).concat(system(TRAIN, {}, [])), full: paramFull,
        cap: 'Таблица Embedding, веса ячейки LSTM, Dense и параметры rmsprop — всё, с чем начинается fit. В книге веса случайные, здесь круглые: любой шаг ниже считается на калькуляторе. Формулы сверху — те, что дальше будут считаться по этим числам.' };
      return m;
    }
    if (at('p = [sigmoid')) {
      const sub = [];
      S.p = S.H.map((h, k) => {
        const z = R(S.w * h + S.b), val = check(`p_${k + 1}`, v.p[k], 1 / (1 + R(Math.exp(-z))));
        sub.push(`p_${k + 1} &= \\sigma(${N(S.w)}\\cdot ${M(h)} + ${M(S.b)}) = ${sigma(z, val)}`);
        return val;
      });
      train.p = ` = (${S.p.map(N).join(',\\ ')})`;
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['p']), sub: aligned(sub), cap: 'Прямой проход до конца: Dense + sigmoid дают вероятность того, что отзыв позитивный.', w: ['sigmoid'] });
    }
    if (at('acc = ')) {
      const y = g.labels, n = S.p.length, guess = S.p.map(p => p > 0.5 ? 1 : 0), hits = guess.filter((q, k) => q === y[k]).length;
      const val = check('acc', v.acc, hits / n); train.acc = ` = ${N(val)}`;
      const sub = S.p.map((p, k) => `p_${k + 1} = ${N(p)} ${guess[k] ? '>' : '\\le'} 0{.}5 &\\Rightarrow ${guess[k]}\\ ${guess[k] === y[k] ? '=' : '\\ne'}\\ y_${k + 1} = ${y[k]}\\ ${guess[k] === y[k] ? '\\checkmark' : '\\times'}`);
      sub.push(`\\mathrm{acc} &= \\tfrac{${hits}}{${n}} = ${N(val)}`);
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['acc']), sub: aligned(sub),
        cap: 'metrics=["acc"]: доля отзывов, угаданных при пороге 0.5 (ровно 0.5 — класс 0). На обучение не влияет — Keras её только печатает.' });
    }
    if (at('loss = bce')) {
      const y = g.labels, n = S.p.length;
      const full = S.p.map((p, k) => `${y[k]}\\cdot\\ln ${N(p)} + ${1 - y[k]}\\cdot\\ln(1-${N(p)})`).join(' + ');
      const kept = S.p.map((p, k) => y[k] ? `\\ln ${N(p)}` : `\\ln ${N(R(1 - p))}`);
      const logs = S.p.map((p, k) => R(Math.log(y[k] ? p : R(1 - p))));
      S.L = check('L', v.loss, -logs.reduce((a, b) => a + b) / n); train.L = ` = ${N(S.L)}`;
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['L']), w: ['logistic-cost'],
        sub: aligned([`L &= -\\tfrac1${n}\\big[${full}\\big]`, `&= -\\tfrac1${n}\\big[${kept.join(' + ')}\\big] = -\\tfrac1${n}\\big[${logs.map(M).join(' + ')}\\big] = ${N(S.L)}`]),
        cap: `ln — натуральный логарифм. y = 1 оставляет ln p, y = 0 — ln(1 − p). N = ${n}: оба отзыва в одном батче. Это число Keras печатает как loss эпохи — оно посчитано до шага обучения.` });
    }
    if (at('print(f"Epoch')) {
      s.k = { title: epochTitle(), full: epochFull, w: ['sigmoid', 'logistic-cost', 'logistic-gradient-component'],
        cap: 'Эпоха целиком: прямой проход, метрика, ошибка, градиент, шаг rmsprop. Строка «Epoch …» печатает acc и loss, посчитанные до шага; новые w и b пойдут в следующую эпоху.' };
      epochFull = [];
      return { title: epochTitle(), rows: system(TRAIN, train, []), cap: 'Та же строка, что печатает Keras: acc и loss этой эпохи.' };
    }
    if (at('print(f"Dense')) { s.k = {}; return { title: 'После fit', rows: [{ t: `w = ${N(S.w)},\\quad b = ${N(S.b)}`, s: 'cur' }], cap: 'Веса Dense после трёх эпох — те же, что выдаёт Keras.' }; }
    if (at('v_w = v_b = ')) { S.v_w = 0; S.v_b = 0; return ep({ title: 'model.fit · перед эпохами', rows: system(TRAIN, {}, []), sub: 'v_w = v_b = 0', cap: 'RMSprop копит средний квадрат градиента; в начале он нулевой.' }); }
    if (at('for epoch')) {
      if (s.x) return null;
      epoch = v.epoch; train = {};
      return { title: epochTitle(), rows: system(TRAIN, train, []), cap: 'Эпоха = прямой проход, ошибка, градиент и один шаг rmsprop на всём батче. Обучаем только Dense; в книге веса Embedding и LSTM меняются так же, через цепное правило.' };
    }
    if (at('g_w = ') || at('g_b = ')) {
      const w = at('g_w'), y = g.labels, n = S.p.length, d = S.p.map((p, k) => R(p - y[k]));
      const parts = w ? d.map((dk, k) => R(dk * S.H[k])) : d;
      const val = check(w ? 'g_w' : 'g_b', w ? v.g_w : v.g_b, parts.reduce((a, b) => a + b) / n);
      S[w ? 'g_w' : 'g_b'] = val; train[w ? 'g_w' : 'g_b'] = ` = ${N(val)}`;
      const terms = S.p.map((p, k) => `(${N(p)} - ${y[k]})` + (w ? `\\cdot ${M(S.H[k])}` : '')).join(' + ');
      return ep({ title: epochTitle(), rows: system(TRAIN, train, [w ? 'g_w' : 'g_b']), w: ['logistic-gradient-component'],
        sub: aligned([`g_${w ? 'w' : 'b'} &= \\tfrac1${n}\\big[${terms}\\big]`, `&= \\tfrac1${n}\\big[${parts.map(M).join(' + ')}\\big] = ${N(val)}`]),
        cap: w ? 'Для пары sigmoid + кросс-энтропия производная по z равна p − y; по w — ещё умножить на вход h.' : 'То же для смещения: его «вход» равен 1.' });
    }
    if (at('v_w = rho') || at('v_b = rho')) {
      const k = at('v_w') ? 'w' : 'b', gk = S['g_' + k], prev = S['v_' + k], g2 = R(gk * gk), rho = S.rho, rest = R(1 - rho);
      const val = check('v_' + k, v['v_' + k], rho * prev + rest * g2); S['v_' + k] = val; train['v_' + k] = ` = ${N(val)}`;
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['v_' + k]),
        sub: `v_${k} = ${N(rho)}\\cdot ${N(prev)} + ${N(rest)}\\cdot ${M(gk)}^2 = ${N(rho)}\\cdot ${N(prev)} + ${N(rest)}\\cdot ${N(g2)} = ${N(val)}`, cap: 'ρ = 0.9 — параметр rmsprop по умолчанию.' });
    }
    if (at('w_d -= ') || at('b_d -= ')) {
      const k = at('w_d') ? 'w' : 'b', old = S[k], gk = S['g_' + k], vk = S['v_' + k];
      const sq = R(Math.sqrt(vk + S.eps)), ratio = R(gk / sq), lr = N(S.lr);
      const val = check(k, v[k + '_d'], old - S.lr * ratio); S[k] = val; train[k] = ` = ${N(val)}`;
      return ep({ title: epochTitle(), rows: system(TRAIN, train, [k]),
        sub: aligned([`${k} &= ${N(old)} - ${lr}\\cdot\\frac{${M(gk)}}{\\sqrt{${N(vk)} + 10^{-7}}} = ${N(old)} - ${lr}\\cdot\\frac{${M(gk)}}{${N(sq)}}`, `&= ${N(old)} - ${lr}\\cdot ${M(ratio)} = ${N(val)}`]),
        cap: 'η = 0.001, ε = 10⁻⁷ — как в Keras 3, ε стоит под корнем. Градиент делится на корень из своего среднего квадрата v, поэтому размер шага почти не зависит от величины g: в первой эпохе v = 0.1·g², и шаг = η/√0.1 ≈ 3.16η = 0.00316 (пока |g| ≫ 0.001). При почти одинаковых g дальше v ≈ 0.19·g², 0.27·g² — шаг ≈ 2.29η, 1.92η, а с ростом числа шагов стремится к η.' });
    }
    return null;
  }

  // v — переменные текущей функции, g — глобальные (внутри lstm_step — с последнего шага модуля).
  steps.forEach(s => {
    const v = s.n, g = s.f ? G : v;
    const m = math(src[s.l[0] - 1].trim(), s, v, g);
    if (m) s.m = m;
    if (!s.f) G = v;
    delete s.n;
  });
  // Последняя строка программы обязана завершать шаг разбора: иначе страница не дойдёт до конца.
  if (!steps[steps.length - 1].k) throw Error('Последняя строка разбора не помечена шагом разбора (s.k)');
  return steps.filter(s => s.k).length;
};
