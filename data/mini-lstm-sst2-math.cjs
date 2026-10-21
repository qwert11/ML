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
const pyList = a => `[${a.map(N).join(', ')}]`;         // repr списка Python: [0.0, 0.0], [-0.8, 0.3]
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

const GATES = [['i', 'i', '\\sigma'], ['f', 'f', '\\sigma'], ['c_hat', '\\tilde c', '\\tanh'], ['o', 'o', '\\sigma']];
const col = sym => sym === '\\tilde c' ? 'c' : sym;
// Порядок как в коде и в Keras: вектор слова x_t умножается на столбец W_i таблицы W (x @ W).
const LSTM = [
  ['x', 'x_t = E[w_t]'],
  ...GATES.map(([key, sym, fn]) => [key, `${sym}_t = ${fn}(x_t W_${col(sym)} + h_{t-1} U_${col(sym)} + b_${col(sym)})`]),
  ['c', 'c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde c_t'],
  ['h', 'h_t = o_t \\odot \\tanh(c_t)']
];
// Порядок как внутри model.fit: прямой проход, метрика, ошибка, градиент, шаг rmsprop. w_d, b_d — вес и смещение Dense.
const TRAIN = [
  ['p', 'p_k = \\sigma(w_d\\,h^{(k)} + b_d)'],
  ['acc', '\\mathrm{acc} = \\frac1N\\sum_k \\big[\\,[p_k > 0{.}5] = y_k\\,\\big]'],
  ['L', 'L = -\\frac1N\\sum_k \\big[y_k \\ln p_k + (1-y_k)\\ln(1-p_k)\\big]'],
  ['g_w', 'g_w = \\frac{\\partial L}{\\partial w_d} = \\frac1N\\sum_k (p_k - y_k)\\,h^{(k)}'],
  ['g_b', 'g_b = \\frac{\\partial L}{\\partial b_d} = \\frac1N\\sum_k (p_k - y_k)'],
  ['v_w', 'v_w \\leftarrow \\rho\\,v_w + (1-\\rho)\\,g_w^2'],
  ['v_b', 'v_b \\leftarrow \\rho\\,v_b + (1-\\rho)\\,g_b^2'],
  ['w', 'w_d \\leftarrow w_d - \\eta\\,\\frac{g_w}{\\sqrt{v_w + \\varepsilon}}'],
  ['b', 'b_d \\leftarrow b_d - \\eta\\,\\frac{g_b}{\\sqrt{v_b + \\varepsilon}}']
];
const table = (cols, rows) => `\\begin{array}{r|${'r'.repeat(cols.length)}} & ${cols.join(' & ')}\\\\ \\hline ${rows.map(([name, vals]) => `${name} & ${vals.map(N).join(' & ')}`).join('\\\\')}\\end{array}`;
const GATE_COLS = ['i', 'f', '\\tilde c', 'o'];
// F — как показать аргумент: N для z вентиля (его печатает программа), N10 для w_d·h + b_d (промежуточное).
const sigma = (z, e, val, F = N) => `\\sigma(${F(z)}) = \\frac{1}{1+e^{${F(0 - z)}}} = \\frac{1}{1+${N10(e)}} = ${N(val)}`;
const aligned = rows => `\\begin{aligned}${rows.join('\\\\')}\\end{aligned}`;

module.exports = function addMath(steps, code, output) {
  const src = code.split('\n');
  let G = {}, review = 0, sentence = '', t = 0, epoch = 0, cell = {};
  const S = { H: [] };       // показанные значения: из них считаются следующие формулы
  const P = {};              // имя → число, как его печатает программа: сверка печати со страницей
  let train = {};            // значения уже посчитанных строк текущей системы
  // Шаги разбора (s.k): данные, параметры, каждое слово, выход LSTM по отзыву, каждая эпоха, итог.
  // На странице такой шаг показывает подстановки всех своих строк разом (full) — слово или эпоху
  // можно пересчитать целиком, не листая по строке. Коллекторы копят подстановки до шага.
  let paramFull = [], wordFull = [], gateRows = [], epochFull = [];
  const param = m => { paramFull.push(m.sub); return m; };
  const ep = m => { epochFull.push(m.sub); return m; };
  // actual — число программы (уже округлено r()), exact — та же формула от показанных чисел,
  // shown — та же формула, но через показанные промежуточные числа (путь калькулятора).
  const check = (what, actual, exact, shown) => {
    const want = N(actual);
    if (N(exact) !== want) throw Error(`Формула ${what}: по формуле ${N(exact)}, в программе ${want}`);
    if (shown !== undefined && N(shown) !== want) throw Error(`Формула ${what}: по показанным промежуточным ${N(shown)}, в программе ${want}`);
    return actual;
  };
  const system = (rows, vals, cur) => rows.map(([key, tex]) => ({ t: tex + (vals[key] || ''), s: cur.includes(key) ? 'cur' : vals[key] ? 'done' : 'todo' }));
  const lstmTitle = () => `model.fit → LSTM · отзыв ${review} «${sentence}»` + (t ? ` · t = ${t}, слово «${cell.word}»` : '');
  const lstm = (cur, extra) => ({ title: lstmTitle(), rows: system(LSTM, cell.vals || {}, cur), ...extra });
  const epochTitle = () => `model.fit · эпоха ${epoch}`;
  // Строка print: формул у неё нет — показываем систему текущей части сети и подпись.
  const printed = (title, rows, cap) => ({ title, rows, cap: cap || 'Строка печатает посчитанное: те же числа, что в формулах выше, — сверьте с панелью «Вывод программы».' });
  const sum = a => a.reduce((acc, x) => acc + x, 0); // как sum() в Python: слагаемое к слагаемому, от нуля
  const hRow = () => `H = (${S.H.map(N).join(',\\ ')})`;

  function math(line, s, v, g) {
    const at = prefix => line.startsWith(prefix);
    if (at('print("train_pad:')) {
      s.k = { title: '1–3. Данные → номера', cap: 'Формул сети здесь нет: словарь, номер каждого слова и выравнивание нулями слева. Результат — в переменных: word2idx и train_pad.' };
      return null;
    }
    if (at('E = ')) return param({ title: 'Embedding: слово → вектор', rows: system([LSTM[0]], {}, ['x']),
      sub: `E = ${table(['x_{t,0}', 'x_{t,1}'], g.E.map((row, k) => [`\\text{${g.words[k]}}`, row]))}`,
      cap: 'Обучаемая таблица: строка на слово. В книге в строке 32 числа, здесь 2. В коде E[idx] — строка с номером слова, в формулах она же x_t.' });
    if (at('W = ')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `W = ${table(GATE_COLS, [['x_{t,0}', g.W[0]], ['x_{t,1}', g.W[1]]])}`, cap: 'Веса от входа x_t: столбец на вентиль. W_i в формулах — столбец i этой таблицы.' });
    if (at('U = ')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `U = ${table(GATE_COLS, [['h_{t-1}', g.U]])}`, cap: 'Веса от выхода прошлого шага h_{t−1}: U_i — столбец i.' });
    if (at('b = ')) return param({ title: 'LSTM-ячейка: параметры', rows: system(LSTM, {}, ['i', 'f', 'c_hat', 'o']),
      sub: `b = ${table(GATE_COLS, [['1', g.b]])}`, cap: 'Смещения вентилей ячейки (b_i, b_f, b_c, b_o). У f смещение 1 — так Keras делает по умолчанию (unit_forget_bias).' });

    if (at('for k, seq')) {
      if (s.x) return { title: 'model.fit → выход LSTM', rows: [{ t: hRow(), s: 'cur' }], cap: 'Отзывы кончились: H собран, по одному числу на отзыв. Дальше — обучение Dense.' };
      review = v.k + 1; t = 0; cell = {}; sentence = g.sentences[v.k];
      return lstm([], { cap: 'Новый отзыв: его слова по очереди пройдут через одну и ту же ячейку.' });
    }
    if (at('print(f"отзыв')) return lstm([], { cap: 'Заголовок отзыва в выводе. Память между отзывами не передаётся: ниже h и c снова станут нулями.' });
    if (at('h, c = 0.0')) { S.h = 0; S.c = 0; return lstm([], { sub: 'h_0 = 0,\\quad c_0 = 0', cap: 'Перед первым словом памяти нет.' }); }
    if (at('for idx')) {
      if (s.x) return { title: `model.fit → выход LSTM · отзыв ${review}`, rows: [{ t: `h_T = h_${t} = ${N(S.h)}`, s: 'cur' }],
        cap: 'Слова кончились. Классификатор получит только последний выход h_T.' };
      t++; cell = { word: g.words[v.idx], idx: v.idx, vals: {} };
      return lstm([], { cap: `Слово ${t} из ${g.max_sen_len}: «${cell.word}», его номер ${v.idx}.` });
    }
    if (at('x = E[idx]')) {
      const x = g.E[cell.idx]; cell.x = x; P.x = x; cell.vals.x = ` = ${pair(x)}`;
      return lstm(['x'], { sub: `x_${t} = E[\\text{${cell.word}}] = E_{${cell.idx}} = ${pair(x)}`, cap: 'Embedding: номер слова выбирает строку таблицы E.' });
    }
    if (at('print(f"{words[idx]')) {
      P.h = S.h; P.c = S.c;
      return lstm(['x'], { sub: `x_${t} = ${pair(cell.x)},\\quad h_{${t - 1}} = ${N(S.h)},\\quad c_{${t - 1}} = ${N(S.c)}`,
        cap: 'Вход шага: вектор слова и память после прошлого слова. Дальше считает lstm_step.' });
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
        // Точная сумма коротких дробей может лечь ровно на …5: тогда она обязана округлиться вверх, как на калькуляторе.
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
      return lstm(['i', 'f', 'c_hat', 'o'], { sub: aligned(rows), cap: 'Сначала аргумент каждого вентиля z: x_t W_i — число за числом вектора слова на столбец i таблицы W (x_{t,0}·W_{0,i} + x_{t,1}·W_{1,i}), плюс h_{t−1} на столбец i таблицы U, плюс смещение. Порядок как в коде и в Keras. Если точная сумма длиннее 6 цифр, она показана перед округлением. Программа печатает z как z_i, z_f, z_c, z_o.' });
    }
    if (at('print(f"   z_i')) return printed(lstmTitle(), system(LSTM, cell.vals, ['i', 'f', 'c_hat', 'o']));
    const gate = GATES.find(([key]) => at(key + ' = r('));
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
      const sub = `c_t = ${N(cell.f)}\\cdot ${M(cell.cp)} + ${N(cell.i)}\\cdot ${M(cell.c_hat)} = ${N10(a)} + ${M10(b)} = ${N(cell.c)}`;
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
    if (at('return h, c')) return lstm([], { sub: `(h_t,\\ c_t) = (${N(cell.h)},\\ ${N(cell.c)})`, cap: 'Шаг t закончен: функция отдаёт выход и память.' });
    if (at('h, c = lstm_step')) {
      S.h = check('h', v.h, cell.h); S.c = check('c', v.c, cell.c);
      s.k = { title: lstmTitle(), full: wordFull, w: ['sigmoid', 'tanh-activation'],
        cap: `Слово «${cell.word}» целиком: вход, аргументы вентилей, вентили, память, выход. Каждое число считается из показанных выше и напечатано программой; h_t и c_t пойдут в следующее слово.` };
      return lstm([], { sub: `h_${t} = ${N(S.h)},\\quad c_${t} = ${N(S.c)}`, cap: 'Для следующего слова это будут h_{t−1} и c_{t−1}.' });
    }
    if (at('H.append')) {
      S.H.push(S.h);
      return { title: 'model.fit → выход LSTM', rows: [{ t: `h^{(${review})} = h_T = ${N(S.h)}`, s: 'cur' }],
        sub: hRow(), cap: 'По одному числу на отзыв — всё, что увидит классификатор. В коде это список H; H[k] в формулах — h^{(k)}.' };
    }
    if (at('print(f"   h_T')) {
      P.h_T = S.h;
      const m = { title: 'model.fit → выход LSTM', rows: [{ t: `h^{(${review})} = h_T = ${N(S.h)}`, s: 'cur' }],
        sub: hRow(), cap: 'По одному числу на отзыв — всё, что увидит классификатор. В коде это список H; H[k] в формулах — h^{(k)}.' };
      s.k = { title: `model.fit → выход LSTM · отзыв ${review} «${sentence}»`, cap: `Слова кончились: классификатор получит только последний выход h_T = ${N(S.h)}. ${m.cap}` };
      return m;
    }
    if (at('w_d, b_d = ')) { S.w = v.w_d; S.b = v.b_d; return param({ title: 'Dense: параметры', rows: system([TRAIN[0]], {}, []), sub: `w_d = ${N(S.w)},\\quad b_d = ${N(S.b)}`, cap: 'Веса слоя Dense(1): один вес на h и смещение. Индекс d — Dense: b без индекса выше — смещения вентилей LSTM. Считать по ним будем внутри fit.' }); }
    if (at('lr, rho, eps = ')) {
      S.lr = v.lr; S.rho = v.rho; S.eps = v.eps;
      const m = param({ title: 'compile: параметры rmsprop', rows: system(TRAIN.slice(5), {}, ['v_w', 'v_b', 'w', 'b']), sub: `\\eta = ${N(S.lr)},\\quad \\rho = ${N(S.rho)},\\quad \\varepsilon = 10^{-7}`,
        cap: 'optimizer="rmsprop" — это ровно эти числа, значения Keras по умолчанию: lr в коде — η, rho — ρ, eps — ε. Сами формулы сработают в каждой эпохе fit.' });
      s.k = { title: '4–5. Модель и compile: все числа до обучения', rows: system(LSTM, {}, []).concat(system(TRAIN, {}, [])), full: paramFull,
        cap: 'Таблица Embedding, веса ячейки LSTM, Dense и параметры rmsprop — всё, с чем начинается fit. В книге веса случайные, здесь круглые: любой шаг ниже считается на калькуляторе. Формулы сверху — те, что дальше будут считаться по этим числам.' };
      return m;
    }
    if (at('v_w = v_b = ')) { S.v_w = 0; S.v_b = 0; return ep({ title: 'model.fit · перед эпохами', rows: system(TRAIN, {}, []), sub: 'v_w = v_b = 0', cap: 'RMSprop копит средний квадрат градиента; в начале он нулевой.' }); }
    if (at('for epoch')) {
      if (s.x) return { title: 'model.fit', rows: system(TRAIN, train, []), cap: 'Эпохи кончились: fit завершён, w_d и b_d — итоговые.' };
      epoch = v.epoch; train = {};
      return { title: epochTitle(), rows: system(TRAIN, train, []), cap: 'Эпоха = прямой проход, ошибка, градиент и один шаг rmsprop на всём батче. Учится только Dense: градиент назад через ячейку LSTM и таблицу E удвоил бы разбор; в книге учатся все слои, тем же правилом цепочки.' };
    }
    if (at('print(f"Epoch')) return printed(epochTitle(), system(TRAIN, train, []), 'Заголовок эпохи — как у Keras. Дальше: прямой проход, метрика, ошибка, градиент, шаг rmsprop.');
    if (at('p = [r(')) {
      const sub = [];
      S.p = S.H.map((h, k) => {
        const z = S.w * h + S.b, eShown = Math.exp(-R10(z));
        const val = check(`p_${k + 1}`, v.p[k], 1 / (1 + Math.exp(-z)), 1 / (1 + R10(eShown)));
        P[`p_${k + 1}`] = val;
        // Две строки на отзыв: с 10-значными числами одна строка не влезает в панель.
        sub.push(`p_${k + 1} &= \\sigma(${N(S.w)}\\cdot ${M(h)} + ${M(S.b)}) = \\sigma(${N10(z)})`, `&= ${sigma(z, eShown, val, N10).replace(/^\\sigma\([^)]*\) = /, '')}`);
        return val;
      });
      train.p = ` = (${S.p.map(N).join(',\\ ')})`;
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['p']), sub: aligned(sub), cap: 'Прямой проход до конца: Dense + sigmoid дают вероятность того, что отзыв позитивный. В коде p — список p_1, p_2.', w: ['sigmoid'] });
    }
    if (at('print(f"   p_1')) return printed(epochTitle(), system(TRAIN, train, ['p']));
    if (at('acc = ')) {
      const y = g.labels, n = S.p.length, guess = S.p.map(p => p > 0.5 ? 1 : 0), hits = guess.filter((q, k) => q === y[k]).length;
      const val = check('acc', v.acc, hits / n); train.acc = ` = ${N(val)}`; P.acc = val;
      const sub = S.p.map((p, k) => `p_${k + 1} = ${N(p)} ${guess[k] ? '>' : '\\le'} 0{.}5 &\\Rightarrow ${guess[k]}\\ ${guess[k] === y[k] ? '=' : '\\ne'}\\ y_${k + 1} = ${y[k]}\\ ${guess[k] === y[k] ? '\\checkmark' : '\\times'}`);
      sub.push(`\\mathrm{acc} &= \\tfrac{${hits}}{${n}} = ${N(val)}`);
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['acc']), sub: aligned(sub),
        cap: 'metrics=["acc"]: доля отзывов, угаданных при пороге 0.5 (ровно 0.5 — класс 0); [условие] в формуле — 1, если верно, иначе 0. На обучение не влияет — Keras её только печатает.' });
    }
    if (at('loss = r(bce')) {
      const y = g.labels, n = S.p.length;
      // Слагаемое к слагаемому, как bce() в программе: y·ln p + (1 − y)·ln(1 − p).
      const terms = S.p.map((p, k) => y[k] * Math.log(p) + (1 - y[k]) * Math.log(1 - p));
      const logs = S.p.map((p, k) => y[k] ? Math.log(p) : Math.log(1 - p)); // что остаётся после умножения на 0 и 1
      S.L = check('L', v.loss, -sum(terms) / n, -sum(logs.map(R10)) / n); train.L = ` = ${N(S.L)}`; P.loss = S.L;
      const full = S.p.map((p, k) => `${y[k]}\\cdot\\ln ${N(p)} + ${1 - y[k]}\\cdot\\ln(1-${N(p)})`).join(' + ');
      const kept = S.p.map((p, k) => y[k] ? `\\ln ${N(p)}` : `\\ln ${N(1 - p)}`);
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['L']), w: ['logistic-cost'],
        sub: aligned([`L &= -\\tfrac1${n}\\big[${full}\\big]`, `&= -\\tfrac1${n}\\big[${kept.join(' + ')}\\big] = -\\tfrac1${n}\\big[${logs.map(M10).join(' + ')}\\big] = ${N(S.L)}`]),
        cap: `ln — натуральный логарифм. y = 1 оставляет ln p, y = 0 — ln(1 − p). N = ${n}: оба отзыва в одном батче. В коде это loss — то число, которое Keras печатает как loss эпохи; оно посчитано до шага обучения.` });
    }
    if (at('print(f"   acc')) return printed(epochTitle(), system(TRAIN, train, ['acc', 'L']));
    if (at('g_w = r(') || at('g_b = r(')) {
      const w = at('g_w'), y = g.labels, n = S.p.length;
      const parts = S.p.map((p, k) => w ? (p - y[k]) * S.H[k] : p - y[k]);
      const val = check(w ? 'g_w' : 'g_b', w ? v.g_w : v.g_b, sum(parts) / n, sum(parts.map(R10)) / n);
      S[w ? 'g_w' : 'g_b'] = val; train[w ? 'g_w' : 'g_b'] = ` = ${N(val)}`; P[w ? 'g_w' : 'g_b'] = val;
      const terms = S.p.map((p, k) => `(${N(p)} - ${y[k]})` + (w ? `\\cdot ${M(S.H[k])}` : '')).join(' + ');
      return ep({ title: epochTitle(), rows: system(TRAIN, train, [w ? 'g_w' : 'g_b']), w: ['logistic-gradient-component'],
        sub: aligned([`g_${w ? 'w' : 'b'} &= \\tfrac1${n}\\big[${terms}\\big]`, `&= \\tfrac1${n}\\big[${parts.map(M10).join(' + ')}\\big] = ${N(val)}`]),
        cap: w ? 'Для пары sigmoid + кросс-энтропия производная слагаемого одного отзыва по аргументу его сигмоиды равна p_k − y_k (вывод — по ссылке ниже); по w_d — ещё умножить на вход h^{(k)}; L — среднее по N отзывам, отсюда 1/N.' : 'То же для смещения b_d: его «вход» равен 1.' });
    }
    if (at('print(f"   g_w')) return printed(epochTitle(), system(TRAIN, train, ['g_w', 'g_b']));
    if (at('v_w = r(rho') || at('v_b = r(rho')) {
      const k = at('v_w') ? 'w' : 'b', gk = S['g_' + k], prev = S['v_' + k], rho = S.rho, rest = 1 - rho, g2 = gk ** 2;
      const val = check('v_' + k, v['v_' + k], rho * prev + rest * g2, rho * prev + R10(rest) * R10(g2)); S['v_' + k] = val; train['v_' + k] = ` = ${N(val)}`; P['v_' + k] = val;
      return ep({ title: epochTitle(), rows: system(TRAIN, train, ['v_' + k]),
        sub: `v_${k} = ${N(rho)}\\cdot ${N(prev)} + ${N(rest)}\\cdot ${M(gk)}^2 = ${N(rho)}\\cdot ${N(prev)} + ${N(rest)}\\cdot ${N10(g2)} = ${N(val)}`, cap: 'ρ = 0.9 — параметр rmsprop по умолчанию: средний квадрат градиента помнит прошлое на 90 %.' });
    }
    if (at('print(f"   v_w')) return printed(epochTitle(), system(TRAIN, train, ['v_w', 'v_b']));
    if (at('w_d = r(') || at('b_d = r(')) {
      const k = at('w_d') ? 'w' : 'b', old = S[k], gk = S['g_' + k], vk = S['v_' + k];
      const sq = Math.sqrt(vk + S.eps), sqShown = R10(sq), ratioShown = gk / sqShown, lr = N(S.lr);
      const val = check(k, v[k + '_d'], old - S.lr * gk / sq, old - S.lr * R10(ratioShown)); S[k] = val; train[k] = ` = ${N(val)}`; P[k + '_d'] = val;
      return ep({ title: epochTitle(), rows: system(TRAIN, train, [k]),
        sub: aligned([`${k}_d &= ${N(old)} - ${lr}\\cdot\\frac{${M(gk)}}{\\sqrt{${N(vk)} + 10^{-7}}} = ${N(old)} - ${lr}\\cdot\\frac{${M(gk)}}{${N10(sqShown)}}`, `&= ${N(old)} - ${lr}\\cdot ${M10(ratioShown)} = ${N(val)}`]),
        cap: k === 'w'
          ? 'η = 0.001, ε = 10⁻⁷ — как в Keras 3, ε стоит под корнем. Градиент делится на корень из своего среднего квадрата v, поэтому размер шага почти не зависит от величины g: в первой эпохе v = 0.1·g², и шаг = η/√0.1 ≈ 3.16η = 0.00316 (пока |g| ≫ 0.001). При почти одинаковых g дальше v ≈ 0.19·g², 0.27·g² — шаг ≈ 2.29η, 1.92η, а с ростом числа шагов стремится к η.'
          : 'То же для смещения b_d, со своими g_b и v_b.' });
    }
    if (at('print(f"   w_d = ')) return printed(epochTitle(), system(TRAIN, train, ['w', 'b']));
    if (at('print(f"1/1')) {
      s.k = { title: epochTitle(), full: epochFull, w: ['sigmoid', 'logistic-cost', 'logistic-gradient-component'],
        cap: 'Эпоха целиком: прямой проход, метрика, ошибка, градиент, шаг rmsprop. Строка «1/1 - acc … loss» — как у Keras: acc и loss посчитаны до шага; новые w_d и b_d пойдут в следующую эпоху.' };
      epochFull = [];
      return { title: epochTitle(), rows: system(TRAIN, train, []), cap: 'Та же строка, что печатает Keras: acc и loss этой эпохи, посчитанные до шага rmsprop.' };
    }
    if (at('print(f"Dense')) { s.k = {}; return { title: 'После fit', rows: [{ t: `w_d = ${N(S.w)},\\quad b_d = ${N(S.b)}`, s: 'cur' }], cap: 'Веса Dense после трёх эпох — те же, что выдаёт Keras (он печатает 4 знака).' }; }
    return null;
  }

  // v — переменные текущей функции, g — глобальные (внутри lstm_step — с последнего шага модуля).
  // После каждого шага всё, что он напечатал как «имя = число», сверяется с числом страницы под этим именем.
  const lines = output.split('\n');
  let seen = 0;
  steps.forEach(s => {
    const v = s.n, g = s.f ? G : v;
    const m = math(src[s.l[0] - 1].trim(), s, v, g);
    if (m) s.m = m;
    if (!s.f) G = v;
    delete s.n;
    for (const line of lines.slice(seen, s.o)) for (const [, name, value] of line.matchAll(/(\S+) = (\[[^\]]*\]|-?\d[\d.e+-]*)/g)) {
      if (!(name in P)) throw Error(`Печать «${name} = ${value}»: у страницы нет такого числа`);
      const want = Array.isArray(P[name]) ? pyList(P[name]) : N(P[name]);
      if (want !== value) throw Error(`Печать «${name} = ${value}»: на странице ${want}`);
    }
    seen = s.o;
  });
  // Последняя строка программы обязана завершать шаг разбора: иначе страница не дойдёт до конца.
  if (!steps[steps.length - 1].k) throw Error('Последняя строка разбора не помечена шагом разбора (s.k)');
  return steps.filter(s => s.k).length;
};
