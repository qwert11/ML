// Мини-версия первого примера книги: та же программа, сокращённая до 2 отзывов.
// Каждый блок = шаг кода Packt → мини-код → его вывод → расчёт формулами.
// Числа в формулах считаются здесь из тех же весов, что и в Python, поэтому
// совпадают с выводом кода. Сборка запускает блоки по очереди и сохраняет вывод каждого.

const E = [[0, 0], [0.2, 0.4], [0.9, 0.1], [0.1, 0.5], [-0.8, 0.3]];
const W = [[0.5, 0.3, 1.0, 0.4], [0.2, -0.1, 0.3, 0.1]];
const U = [0.3, 0.2, 0.5, 0.2];
const B = [0, 1, 0, 0];
const words = ['PAD', 'very', 'good', 'movie', 'bad'];
const pad = [[1, 2, 3], [0, 4, 3]];
const y = [1, 0];

const sig = z => 1 / (1 + Math.exp(-z));
// Число в LaTeX: до 4 знаков, без хвостовых нулей, десятичная запятая.
const n = (v, d = 4) => { let s = v.toFixed(d).replace(/\.?0+$/, ''); if (s === '-0') s = '0'; return s.replace('.', '{,}'); };
const m = v => v < 0 ? `(${n(v)})` : n(v);
const sub = k => String(k).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]);
const sigTex = z => `\\sigma(${n(z)})=\\frac{1}{1+e^{${n(-z)}}}`;

function lstm(seq) {
  let h = 0, c = 0;
  return seq.map(idx => {
    const x = E[idx], hp = h, cp = c;
    const z = [0, 1, 2, 3].map(k => x[0] * W[0][k] + x[1] * W[1][k] + h * U[k] + B[k]);
    const [i, f, ch, o] = [sig(z[0]), sig(z[1]), Math.tanh(z[2]), sig(z[3])];
    c = f * c + i * ch; h = o * Math.tanh(c);
    return { idx, x, hp, cp, z, i, f, ch, o, c, h };
  });
}
const run1 = lstm(pad[0]), run2 = lstm(pad[1]);
const H = [run1[2].h, run2[2].h];
let wd = 2, bd = 0;
const p = H.map(h => sig(wd * h + bd));
const bce = ps => -(ps.map((pk, k) => y[k] * Math.log(pk) + (1 - y[k]) * Math.log(1 - pk)).reduce((a, b) => a + b) / 2);
const L = bce(p);
const gw = ((p[0] - y[0]) * H[0] + (p[1] - y[1]) * H[1]) / 2, gb = ((p[0] - y[0]) + (p[1] - y[1])) / 2;
const vw = 0.1 * gw ** 2, vb = 0.1 * gb ** 2;
const wd2 = wd - 0.001 * gw / (Math.sqrt(vw) + 1e-7), bd2 = bd - 0.001 * gb / (Math.sqrt(vb) + 1e-7);
const p2 = H.map(h => sig(wd2 * h + bd2)), L2 = bce(p2);

const gateNames = ['i', 'f', '\\tilde c', 'o'];
// Полная подстановка одного шага LSTM: всё, что нужно набрать на калькуляторе.
function stepLines(s, t, cPrevName = `c_{${t - 1}}`) {
  const zRows = gateNames.map((g, k) =>
    `z_{${g}} &= ${n(s.x[0])}\\cdot ${m(W[0][k])} + ${n(s.x[1])}\\cdot ${m(W[1][k])} + ${m(s.hp)}\\cdot ${n(U[k])} + ${n(B[k])} = ${n(s.z[k])}`);
  return [
    { latex: `\\begin{aligned}${zRows.join('\\\\')}\\end{aligned}`,
      comment: `x${sub(t)} = (${n(s.x[0]).replace('{,}', ',')}; ${n(s.x[1]).replace('{,}', ',')}) — строка E слова «${words[s.idx]}», h${sub(t - 1)} = ${n(s.hp).replace('{,}', ',')} — выход прошлого шага. Каждая строка: x[0]·W[0] + x[1]·W[1] + h·U + b своего столбца` },
    { latex: `\\begin{aligned}i &= ${sigTex(s.z[0])} = ${n(s.i)}\\\\f &= ${sigTex(s.z[1])} = ${n(s.f)}\\\\\\tilde c &= \\tanh(${n(s.z[2])}) = ${n(s.ch)}\\\\o &= ${sigTex(s.z[3])} = ${n(s.o)}\\end{aligned}`,
      comment: 'σ — «кран» от 0 до 1, tanh — содержимое от −1 до 1. На калькуляторе: e^x и tanh' },
    { latex: `c_${t} = f\\cdot ${cPrevName} + i\\cdot\\tilde c = ${n(s.f)}\\cdot ${m(s.cp)} + ${n(s.i)}\\cdot ${m(s.ch)} = ${n(s.c)}`,
      comment: 'сколько старой памяти оставить + сколько нового записать' },
    { latex: `h_${t} = o\\cdot\\tanh(c_${t}) = ${n(s.o)}\\cdot\\tanh(${n(s.c)}) = ${n(s.o)}\\cdot ${m(Math.tanh(s.c))} = ${n(s.h)}`,
      comment: 'то, что ячейка отдаёт наружу и на следующий шаг' }
  ];
}

const blocks = [
  {
    title: 'Данные: 2 отзыва вместо 67 349',
    book: 'df = pd.read_csv("SST-2/train.tsv", sep="\\t")\nsentences = df.sentence\nlabels = df.label',
    code: `import numpy as np

# В книге 67 349 отзывов из SST-2/train.tsv, здесь — 2
sentences = ["very good movie", "bad movie"]
labels = np.array([1, 0])   # 1 — позитив, 0 — негатив`,
    math: [
      { latex: '\\text{very good movie}\\;\\to\\; y_1 = 1,\\qquad \\text{bad movie}\\;\\to\\; y_2 = 0', comment: 'метка — правильный ответ, к которому обучение будет подтягивать выход сети' }
    ]
  },
  {
    title: 'Словарь: слово → номер',
    book: 'max_sen_len = max([len(s.split()) for s in sentences])\nwords = ["PAD"] + list(set([w for s in sentences for w in s.split()]))\nword2idx = {w: i for i, w in enumerate(words)}\nmax_words = max(word2idx.values()) + 1',
    code: `max_sen_len = max(len(s.split()) for s in sentences)
# В книге порядок задаёт set() и меняется от запуска к запуску;
# здесь он зафиксирован, чтобы номера слов совпадали с разбором
words = ["PAD", "very", "good", "movie", "bad"]
word2idx = {w: i for i, w in enumerate(words)}
max_words = max(word2idx.values()) + 1
print("word2idx =", word2idx)
print("max_sen_len =", max_sen_len, "max_words =", max_words)`,
    math: [
      { latex: '\\text{max\\_sen\\_len} = \\max(3,\\;2) = 3', comment: 'длина самого длинного предложения в словах; в книге это 52' },
      { latex: '\\text{PAD}=0,\\;\\text{very}=1,\\;\\text{good}=2,\\;\\text{movie}=3,\\;\\text{bad}=4\\qquad \\text{max\\_words} = 4 + 1 = 5', comment: '0 зарезервирован под PAD — «пустое место»; max_words — число строк будущей таблицы Embedding' }
    ]
  },
  {
    title: 'Номера слов и выравнивание длины',
    book: 'train = [list(map(lambda x: word2idx[x], s.split())) for s in sentences]\ntrain_pad = preprocessing.sequence.pad_sequences(train, maxlen=max_sen_len)',
    code: `train = [[word2idx[w] for w in s.split()] for s in sentences]
# pad_sequences по умолчанию (padding="pre") дописывает нули СЛЕВА
train_pad = np.array([[0] * (max_sen_len - len(t)) + t for t in train])
print("train =", train)
print("train_pad =", train_pad.tolist())`,
    math: [
      { latex: '\\text{very good movie}\\to[1,\\,2,\\,3]', comment: 'уже длина 3 — дописывать нечего' },
      { latex: '\\text{bad movie}\\to[4,\\,3]\\to[\\,0,\\,4,\\,3\\,]', comment: 'один ноль слева до длины max_sen_len = 3' }
    ],
    note: 'Нули слева, а не справа: LSTM отдаёт состояние после последнего шага, и настоящие слова оказываются ближе к выходу.'
  },
  {
    title: 'Embedding: номер слова → вектор',
    book: 'model.add(Embedding(max_words, 32))',
    code: `# Embedding(max_words, 2): таблица 5 × 2 — по строке на слово.
# Keras заполняет её случайно, здесь — круглые числа
E = np.array([[ 0.0, 0.0],   # 0 PAD
              [ 0.2, 0.4],   # 1 very
              [ 0.9, 0.1],   # 2 good
              [ 0.1, 0.5],   # 3 movie
              [-0.8, 0.3]])  # 4 bad
X = E[train_pad[0]]   # строки 1, 2, 3 — «very good movie»
print("X =", X.tolist())`,
    math: [
      { latex: 'E=\\begin{pmatrix}0 & 0\\\\0{,}2 & 0{,}4\\\\0{,}9 & 0{,}1\\\\0{,}1 & 0{,}5\\\\-0{,}8 & 0{,}3\\end{pmatrix}\\begin{matrix}\\leftarrow\\text{PAD}\\\\\\leftarrow\\text{very}\\\\\\leftarrow\\text{good}\\\\\\leftarrow\\text{movie}\\\\\\leftarrow\\text{bad}\\end{matrix}', comment: 'обучаемые веса слоя: в книге max_words строк × 32 столбца' },
      { latex: 'x_1=E_{[1]}=(0{,}2;\\;0{,}4),\\quad x_2=E_{[2]}=(0{,}9;\\;0{,}1),\\quad x_3=E_{[3]}=(0{,}1;\\;0{,}5)', comment: 'считать нечего: слой просто берёт строку с номером слова' }
    ],
    note: 'Всё, что модель «знает» о слове, — эти числа. Обучение двигает строки E так, чтобы слова с похожей ролью получали похожие векторы.'
  },
  {
    title: 'LSTM: веса и формулы одной ячейки',
    book: 'model.add(LSTM(32))',
    code: `# LSTM(1): одна ячейка. Столбцы — 4 вентиля в порядке Keras:
#                i     f     c~    o
W = np.array([[0.5,  0.3,  1.0,  0.4],    # вес от x[0]
              [0.2, -0.1,  0.3,  0.1]])   # вес от x[1]
U = np.array( [0.3,  0.2,  0.5,  0.2])    # вес от h прошлого шага
b = np.array( [0.0,  1.0,  0.0,  0.0])    # у f смещение 1 (unit_forget_bias)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lstm_step(x, h, c):
    z = x @ W + h * U + b    # z = x0*W[0] + x1*W[1] + h*U + b -> 4 числа
    i = sigmoid(z[0])        # вход: сколько нового записать
    f = sigmoid(z[1])        # забывание: сколько старой памяти оставить
    c_hat = np.tanh(z[2])    # кандидат: что записать (от -1 до 1)
    o = sigmoid(z[3])        # выход: сколько памяти показать наружу
    c = f * c + i * c_hat    # новая память
    h = o * np.tanh(c)       # выход ячейки
    print("  z =", np.round(z, 4).tolist())
    print(f"  i={i:.4f} f={f:.4f} c_hat={c_hat:.4f} o={o:.4f}")
    print(f"  c={c:.4f} h={h:.4f}")
    return h, c`,
    math: [
      { latex: 'z_t = x_t W + h_{t-1} U + b', comment: 'одно умножение даёт сразу 4 числа — по одному на вентиль [i, f, c̃, o]' },
      { latex: 'i,f,o=\\sigma(z)=\\frac{1}{1+e^{-z}},\\qquad \\tilde c=\\tanh(z)', comment: 'три «крана» и кандидат в память' },
      { latex: 'c_t=f\\cdot c_{t-1}+i\\cdot\\tilde c,\\qquad h_t=o\\cdot\\tanh(c_t)', comment: 'c — долгая память ячейки, h — то, что она показывает наружу' },
      { latex: '4\\cdot(2+1+1)=16\\;\\text{весов}\\qquad\\text{в книге: }4\\cdot(32+32+1)\\cdot 32=8\\,320', comment: 'на каждый вентиль: веса от входа + вес от h + смещение' }
    ],
    note: 'Функция только описывает расчёт; ниже она вызывается по разу на каждое слово — так же LSTM(32) проходит предложение слева направо.',
    formulas: ['sigmoid', 'tanh-activation']
  },
  ...[0, 1, 2].map(k => ({
    title: `LSTM, шаг t = ${k + 1}: «${words[pad[0][k]]}»`,
    book: `model.add(LSTM(32))   # тот же слой, ${k ? 'следующее' : 'первое'} слово`,
    code: k === 0
      ? `h, c = 0.0, 0.0   # до первого слова памяти нет
print("t=1 very")
h, c = lstm_step(X[0], h, c)`
      : `print("t=${k + 1} ${words[pad[0][k]]}")
h, c = lstm_step(X[${k}], h, c)`,
    math: stepLines(run1[k], k + 1),
    note: [
      'Памяти ещё нет: f·c₀ = 0, и c₁ целиком набирается из входа. «very» почти нейтрально — h₁ маленький.',
      `«good» даёт крупный кандидат c̃ = ${n(run1[1].ch).replace('{,}', ',')}. Старая память сохраняется на ${Math.round(run1[1].f * 100)} %, и c вырастает в ${Math.round(run1[1].c / run1[0].c)} раза: сеть «запомнила» позитив.`,
      `«movie» нейтрально: кандидат слабый, f держит ${Math.round(run1[2].f * 100)} % памяти, h почти не меняется — оценка «good» дожила до конца предложения. h₃ = ${n(run1[2].h).replace('{,}', ',')} и есть выход LSTM для этого отзыва.`
    ][k]
  })),
  {
    title: 'Второй отзыв «bad movie»: те же веса, другие строки E',
    book: 'model.add(LSTM(32))   # следующий отзыв батча',
    code: `h2, c2 = 0.0, 0.0
for t, idx in enumerate(train_pad[1], 1):   # [0, 4, 3]
    print(f"t={t} {words[idx]}")
    h2, c2 = lstm_step(E[idx], h2, c2)

H = np.array([h, h2])   # LSTM отдаёт только последнее h каждого отзыва
print("H =", np.round(H, 4).tolist())`,
    math: [
      { latex: `t=1\\;\\text{PAD}:\\; x=(0;0)\\Rightarrow z=b,\\;\\; \\tilde c=\\tanh(0)=0,\\;\\; c_1=0,\\;\\; h_1=0`, comment: 'нулевой вектор ничего не записывает в память' },
      ...stepLines(run2[1], 2),
      ...stepLines(run2[2], 3),
      { latex: `H = (h^{(1)}_3;\\; h^{(2)}_3) = (${n(H[0])};\\; ${n(H[1])})`, comment: 'по одному числу на отзыв — дальше классификатор видит только их' }
    ],
    note: 'В настоящем Keras строка PAD случайная и не замаскирована (mask_zero=False), так что нули слева слегка «шумят». «bad» даёт отрицательный кандидат, и h уходит в минус.'
  },
  {
    title: 'Dense + sigmoid: состояние → вероятность',
    book: 'model.add(Dense(1, activation="sigmoid"))',
    code: `w_d, b_d = 2.0, 0.0   # Dense(1): один вес на h и смещение
p = sigmoid(w_d * H + b_d)
print("p =", np.round(p, 4).tolist())`,
    math: [
      { latex: `p_1 = \\sigma(w\\cdot h + b) = \\sigma(2\\cdot ${n(H[0])} + 0) = ${sigTex(wd * H[0])} = ${n(p[0])}`, comment: '«very good movie», y = 1' },
      { latex: `p_2 = \\sigma(2\\cdot ${m(H[1])} + 0) = ${sigTex(wd * H[1])} = ${n(p[1])}`, comment: '«bad movie», y = 0' }
    ],
    note: 'В книге h — вектор из 32 чисел, и Dense берёт их взвешенную сумму w₁h₁ + … + w₃₂h₃₂ + b. Здесь слагаемое одно.',
    formulas: ['sigmoid']
  },
  {
    title: 'Ошибка: binary_crossentropy',
    book: 'model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])',
    code: `y = labels
L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
acc = np.mean((p > 0.5) == y)
print(f"loss = {L:.4f}  acc = {acc}")`,
    math: [
      { latex: 'L=-\\frac1N\\sum_k\\big[y_k\\ln p_k+(1-y_k)\\ln(1-p_k)\\big]', comment: 'среднее по батчу; у нас N = 2 — оба отзыва в одном батче' },
      { latex: `L=-\\tfrac12\\big[1\\cdot\\ln ${n(p[0])} + (1-0)\\cdot\\ln(1-${n(p[1])})\\big]=-\\tfrac12\\big[${n(Math.log(p[0]))} + (${n(Math.log(1 - p[1]))})\\big]=${n(L)}`, comment: 'для y = 1 работает только ln p, для y = 0 — только ln(1 − p)' },
      { latex: `\\text{acc}: ${n(p[0])} > 0{,}5\\to 1 = y_1,\\;\\; ${n(p[1])} < 0{,}5\\to 0 = y_2\\;\\Rightarrow\\;\\text{acc} = 2/2 = 1`, comment: 'метрика acc из compile: доля отзывов, где порог 0,5 дал верный класс' }
    ],
    note: 'Сеть, которая всегда отвечает p = 0,5, получила бы L = ln 2 ≈ 0,693. У нас уже меньше: круглые веса подобраны так, что «good» тянет вверх, а «bad» вниз.',
    formulas: ['logistic-cost']
  },
  {
    title: 'Градиент: куда двигать веса',
    book: 'history = model.fit(train_pad, labels, epochs=30, batch_size=32, validation_split=0.2)',
    code: `# Внутри fit TensorFlow считает производные ВСЕХ весов (E, W, U, b, Dense).
# Здесь — только Dense: для пары sigmoid + binary_crossentropy dL/dz = p - y
g_w = np.mean((p - y) * H)
g_b = np.mean(p - y)
print(f"g_w = {g_w:.4f}  g_b = {g_b:.4f}")`,
    math: [
      { latex: '\\frac{\\partial L}{\\partial w}=\\frac1N\\sum_k (p_k-y_k)\\,h_k,\\qquad \\frac{\\partial L}{\\partial b}=\\frac1N\\sum_k (p_k-y_k)', comment: 'производные сигмоиды и логарифма сокращаются в «ошибка × вход»' },
      { latex: `g_w=\\tfrac12\\big[(${n(p[0])}-1)\\cdot ${n(H[0])}+(${n(p[1])}-0)\\cdot ${m(H[1])}\\big]=\\tfrac12\\big[${m((p[0] - 1) * H[0])}+${m(p[1] * H[1])}\\big]=${n(gw)}`, comment: 'минус → w надо увеличить' },
      { latex: `g_b=\\tfrac12\\big[(${n(p[0])}-1)+(${n(p[1])}-0)\\big]=\\tfrac12\\big[${m(p[0] - 1)}+${n(p[1])}\\big]=${n(gb)}`, comment: 'плюс → b надо уменьшить' }
    ],
    note: 'Градиенты весов LSTM считаются тем же правилом цепочки, только через все шаги t назад (backpropagation through time). Мини-код ограничен Dense, чтобы каждое число оставалось проверяемым.',
    formulas: ['logistic-gradient-component']
  },
  {
    title: 'RMSprop: один шаг обучения',
    book: 'model.compile(optimizer="rmsprop", ...)   # шаг выполняется внутри fit',
    code: `lr, rho, eps = 0.001, 0.9, 1e-7          # параметры rmsprop по умолчанию
v_w = rho * 0 + (1 - rho) * g_w ** 2     # среднее квадрата градиента; было 0
v_b = rho * 0 + (1 - rho) * g_b ** 2
w_d = w_d - lr * g_w / (np.sqrt(v_w) + eps)
b_d = b_d - lr * g_b / (np.sqrt(v_b) + eps)

p = sigmoid(w_d * H + b_d)               # снова прямой проход
L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"w = {w_d:.4f}  b = {b_d:.4f}")
print("p =", np.round(p, 4).tolist(), f" loss = {L:.4f}")`,
    math: [
      { latex: 'v\\leftarrow\\rho\\cdot v+(1-\\rho)\\cdot g^2,\\qquad w\\leftarrow w-\\eta\\cdot\\frac{g}{\\sqrt v+\\varepsilon}', comment: 'η = 0,001, ρ = 0,9; ε = 10⁻⁷ на калькуляторе можно не учитывать' },
      { latex: `v_w = 0{,}9\\cdot 0 + 0{,}1\\cdot ${m(gw)}^2 = 0{,}1\\cdot g_w^2,\\qquad v_b = 0{,}1\\cdot g_b^2`, comment: 'на первом шаге v ещё пустое, поэтому v = 0,1·g²' },
      { latex: '\\frac{g}{\\sqrt{v}} = \\frac{g}{\\sqrt{0{,}1}\\cdot|g|} = \\pm\\frac{1}{\\sqrt{0{,}1}} = \\pm 3{,}1623', comment: 'величина градиента сокращается — остаётся только знак (так считать точнее, чем делить округлённые g и √v)' },
      { latex: `w = 2 - 0{,}001\\cdot(-3{,}1623) = ${n(wd2)},\\qquad b = 0 - 0{,}001\\cdot 3{,}1623 = ${n(bd2)}`, comment: 'g_w < 0 → w растёт, g_b > 0 → b падает; оба сдвига ≈ 0,0032' },
      { latex: `p = (${n(p2[0])};\\; ${n(p2[1])}),\\qquad L = ${n(L2)} < ${n(L)}`, comment: 'после шага: p₁ подросла, p₂ снизилась, ошибка уменьшилась' }
    ],
    note: 'В книге таких шагов 30 эпох × ⌈53 879 / 32⌉ = 30 × 1 684 = 50 520 (validation_split=0.2 откладывает пятую часть из 67 349 отзывов). plot_graphs только рисует history — расчёта в нём нет.'
  }
];

module.exports = {
  intro: 'Та же программа, что в книге, ужатая до листка бумаги: 2 отзыва, словарь из 5 слов, вектор слова из 2 чисел и LSTM с одной ячейкой. Веса круглые вместо случайных. Код разбит на те же шаги: сверху строка книги, затем мини-код с комментариями, его вывод и тот же расчёт формулами с подставленными числами — можно пересчитать на калькуляторе. С этими весами настоящий Keras выдаёт те же вероятности 0,6509 и 0,4792.',
  rounding: 'Числа округлены до 4 знаков только при показе — код считает точно. Если пересчитывать с округлёнными значениями, последний знак может отличаться на 1.',
  scale: [
    ['Отзывы', '67 349', '2'],
    ['max_words', 'весь словарь SST-2', '5'],
    ['Embedding', '32 числа на слово', '2'],
    ['LSTM', '32 ячейки', '1'],
    ['Обучение', '30 эпох, batch_size=32', '1 шаг RMSprop, только Dense'],
    ['validation_split, графики', 'есть', 'пропущены — не влияют на расчёт']
  ],
  blocks
};
