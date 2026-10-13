// Мини-разбор первого примера книги: те же слои на числах, которые можно проверить вручную.
// Шаги сгенерированы из этих весов; результат совпадает с Keras при set_weights.
module.exports = {
  "intro": "Та же модель, что в книге, но в масштабе, который помещается на листок: 2 предложения, словарь из 5 слов, вектор слова из 2 чисел и LSTM с одной ячейкой. Веса заданы круглыми вместо случайных, поэтому каждое число ниже можно пересчитать на калькуляторе. С теми же весами настоящий Keras (3.15) выдаёт те же вероятности 0,6509 и 0,4792.",
  "scale": [
    [
      "Предложения",
      "67 349",
      "2"
    ],
    [
      "max_words",
      "весь словарь SST-2",
      "5"
    ],
    [
      "Embedding",
      "32 числа на слово",
      "2"
    ],
    [
      "LSTM",
      "32 ячейки",
      "1"
    ],
    [
      "batch_size",
      "32",
      "2"
    ],
    [
      "Шаги обучения",
      "30 эпох",
      "3 шага, только Dense"
    ]
  ],
  "code": "import numpy as np\n\n# Мини-версия примера книги: 2 предложения, словарь из 5 слов,\n# Embedding(5, 2) → LSTM(1) → Dense(1, sigmoid). Всё считается вручную.\n\nsentences = [\"very good movie\", \"bad movie\"]\nlabels = np.array([1, 0])\n\n# В книге порядок слов задаёт set(); здесь фиксируем его, чтобы индексы не прыгали\nwords = [\"PAD\", \"very\", \"good\", \"movie\", \"bad\"]\nword2idx = {w: i for i, w in enumerate(words)}\nmax_words = len(words)                                   # 5\nmax_sen_len = max(len(s.split()) for s in sentences)     # 3\n\ntrain = [[word2idx[w] for w in s.split()] for s in sentences]\n# pad_sequences(padding=\"pre\"): нули дописываются слева\ntrain_pad = np.array([[0] * (max_sen_len - len(t)) + t for t in train])\nprint(\"train_pad =\", train_pad.tolist())\n\n# Keras начинает со случайных весов; здесь — круглые, чтобы считать на бумаге\nE = np.array([[0.0, 0.0],    # PAD\n              [0.2, 0.4],    # very\n              [0.9, 0.1],    # good\n              [0.1, 0.5],    # movie\n              [-0.8, 0.3]])  # bad          Embedding(max_words, 2)\nW = np.array([[0.5, 0.3, 1.0, 0.4],\n              [0.2, -0.1, 0.3, 0.1]])  # LSTM kernel: x → вентили [i, f, c̃, o]\nU = np.array([0.3, 0.2, 0.5, 0.2])     # recurrent_kernel: h → [i, f, c̃, o]\nb = np.array([0.0, 1.0, 0.0, 0.0])     # unit_forget_bias=True: у f смещение 1\nw_d, b_d = 2.0, 0.0                    # Dense(1, activation=\"sigmoid\")\n\nsigmoid = lambda z: 1 / (1 + np.exp(-z))\n\ndef lstm(seq, show=False):\n    h = c = 0.0\n    for t, idx in enumerate(seq, 1):\n        z = E[idx] @ W + h * U + b\n        i, f, o = sigmoid(z[0]), sigmoid(z[1]), sigmoid(z[3])\n        c_hat = np.tanh(z[2])\n        c = f * c + i * c_hat\n        h = o * np.tanh(c)\n        if show:\n            print(f\"  t={t} {words[idx]:5} z={np.round(z, 3).tolist()} \"\n                  f\"i={i:.3f} f={f:.3f} c̃={c_hat:.3f} o={o:.3f} → c={c:.3f} h={h:.3f}\")\n    return h\n\n# Прямой проход\nH = []\nfor s, seq in zip(sentences, train_pad):\n    print(f'\"{s}\" {seq.tolist()}')\n    H.append(lstm(seq, show=True))\nH = np.array(H)\n\n# model.compile(optimizer=\"rmsprop\", loss=\"binary_crossentropy\"); fit(batch_size=2)\nlr, rho, eps = 0.001, 0.9, 1e-7\nv_w = v_b = 0.0\nfor step in range(1, 4):\n    p = sigmoid(w_d * H + b_d)\n    loss = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))\n    g_w = np.mean((p - labels) * H)     # ∂L/∂w для пары sigmoid + BCE\n    g_b = np.mean(p - labels)\n    v_w = rho * v_w + (1 - rho) * g_w ** 2\n    v_b = rho * v_b + (1 - rho) * g_b ** 2\n    w_d -= lr * g_w / (np.sqrt(v_w) + eps)\n    b_d -= lr * g_b / (np.sqrt(v_b) + eps)\n    print(f\"шаг {step}: p={np.round(p, 4).tolist()} loss={loss:.4f} \"\n          f\"→ w={w_d:.4f} b={b_d:.4f}\")",
  "steps": [
    {
      "title": "Словарь и выравнивание",
      "code": "train_pad = [[0] * (max_sen_len - len(t)) + t for t in train]",
      "lines": [
        {
          "latex": "\\text{PAD}{=}0,\\;\\text{very}{=}1,\\;\\text{good}{=}2,\\;\\text{movie}{=}3,\\;\\text{bad}{=}4",
          "comment": "word2idx: каждому слову — номер строки в таблице эмбеддингов; 0 зарезервирован под PAD"
        },
        {
          "latex": "\\text{very good movie}\\to[1,2,3],\\qquad \\text{bad movie}\\to[0,4,3]",
          "comment": "pad_sequences по умолчанию дописывает нули слева (padding=\"pre\"), чтобы все строки имели длину max_sen_len = 3"
        }
      ],
      "note": "Нули слева — не случайность: LSTM отдаёт состояние после последнего шага, и настоящие слова оказываются ближе к выходу."
    },
    {
      "title": "Embedding: номер слова → вектор",
      "code": "model.add(Embedding(max_words, 32))   # у нас Embedding(5, 2)",
      "lines": [
        {
          "latex": "E=\\begin{pmatrix}0 & 0\\\\0{,}2 & 0{,}4\\\\0{,}9 & 0{,}1\\\\0{,}1 & 0{,}5\\\\-0{,}8 & 0{,}3\\end{pmatrix}\\begin{matrix}\\leftarrow\\text{PAD}\\\\\\leftarrow\\text{very}\\\\\\leftarrow\\text{good}\\\\\\leftarrow\\text{movie}\\\\\\leftarrow\\text{bad}\\end{matrix}",
          "comment": "обучаемая таблица max_words × размер вектора; в книге строк — по числу слов SST-2, столбцов 32"
        },
        {
          "latex": "x_1=E_{[1]}=(0{,}2;\\;0{,}4),\\quad x_2=E_{[2]}=(0{,}9;\\;0{,}1),\\quad x_3=E_{[3]}=(0{,}1;\\;0{,}5)",
          "comment": "умножения нет: слой просто берёт строку с нужным номером (то же, что one-hot × E)"
        }
      ],
      "note": "Всё, что модель «знает» о слове, — эти два числа. Обучение двигает строки E так, чтобы слова с похожей ролью получали похожие векторы."
    },
    {
      "title": "Веса LSTM: четыре вентиля одной операцией",
      "code": "model.add(LSTM(32))   # у нас LSTM(1)",
      "lines": [
        {
          "latex": "z_t = x_t W + h_{t-1} U + b",
          "comment": "одно умножение даёт сразу 4 числа — по одному на вентиль [i, f, c̃, o] (порядок Keras)"
        },
        {
          "latex": "W=\\begin{pmatrix}0{,}5 & 0{,}3 & 1 & 0{,}4\\\\0{,}2 & -0{,}1 & 0{,}3 & 0{,}1\\end{pmatrix},\\quad U=(0{,}3;\\;0{,}2;\\;0{,}5;\\;0{,}2),\\quad b=(0;\\;1;\\;0;\\;0)",
          "comment": "у f смещение 1: Keras ставит unit_forget_bias=True, чтобы в начале обучения ячейка скорее помнила, чем забывала"
        },
        {
          "latex": "i,f,o=\\sigma(z),\\quad \\tilde c=\\tanh(z),\\quad c_t=f\\,c_{t-1}+i\\,\\tilde c,\\quad h_t=o\\tanh(c_t)",
          "comment": "σ даёт «краны» от 0 до 1, tanh — содержимое от −1 до 1; c — долгая память, h — то, что ячейка показывает наружу"
        }
      ],
      "note": "Параметров: 4·(2 + 1 + 1) = 16. В книге та же формула даёт 4·(32 + 32 + 1) = 8 320."
    },
    {
      "title": "LSTM, шаг t = 1: «very»",
      "code": "z = E[idx] @ W + h * U + b",
      "lines": [
        {
          "latex": "\\begin{aligned}z_{i} &= 0{,}2\\cdot 0{,}5 + 0{,}4\\cdot 0{,}2 + 0\\cdot 0{,}3 + 0 = 0{,}18\\\\z_{f} &= 0{,}2\\cdot 0{,}3 + 0{,}4\\cdot (-0{,}1) + 0\\cdot 0{,}2 + 1 = 1{,}02\\\\z_{\\tilde c} &= 0{,}2\\cdot 1 + 0{,}4\\cdot 0{,}3 + 0\\cdot 0{,}5 + 0 = 0{,}32\\\\z_{o} &= 0{,}2\\cdot 0{,}4 + 0{,}4\\cdot 0{,}1 + 0\\cdot 0{,}2 + 0 = 0{,}12\\end{aligned}",
          "comment": "вход x1 = (0,2; 0,4), прошлое h = 0"
        },
        {
          "latex": "i=\\sigma(0{,}18)=0{,}545,\\;\\; f=\\sigma(1{,}02)=0{,}735,\\;\\; \\tilde c=\\tanh(0{,}32)=0{,}31,\\;\\; o=\\sigma(0{,}12)=0{,}53",
          "comment": "вход, забывание, кандидат, выход"
        },
        {
          "latex": "c_1=f\\,c_0+i\\,\\tilde c=0{,}735\\cdot 0+0{,}545\\cdot 0{,}31=0{,}169",
          "comment": "сколько старой памяти оставить + сколько нового записать"
        },
        {
          "latex": "h_1=o\\tanh(c_1)=0{,}53\\cdot\\tanh(0{,}169)=0{,}089",
          "comment": "что ячейка выдаёт на этом шаге"
        }
      ],
      "note": "Памяти ещё нет: f·c₀ = 0, и c₁ целиком набирается из входа. «very» почти нейтрально — h₁ маленький."
    },
    {
      "title": "LSTM, шаг t = 2: «good»",
      "code": "z = E[idx] @ W + h * U + b",
      "lines": [
        {
          "latex": "\\begin{aligned}z_{i} &= 0{,}9\\cdot 0{,}5 + 0{,}1\\cdot 0{,}2 + 0{,}089\\cdot 0{,}3 + 0 = 0{,}497\\\\z_{f} &= 0{,}9\\cdot 0{,}3 + 0{,}1\\cdot (-0{,}1) + 0{,}089\\cdot 0{,}2 + 1 = 1{,}278\\\\z_{\\tilde c} &= 0{,}9\\cdot 1 + 0{,}1\\cdot 0{,}3 + 0{,}089\\cdot 0{,}5 + 0 = 0{,}974\\\\z_{o} &= 0{,}9\\cdot 0{,}4 + 0{,}1\\cdot 0{,}1 + 0{,}089\\cdot 0{,}2 + 0 = 0{,}388\\end{aligned}",
          "comment": "вход x2 = (0,9; 0,1), прошлое h = 0,089"
        },
        {
          "latex": "i=\\sigma(0{,}497)=0{,}622,\\;\\; f=\\sigma(1{,}278)=0{,}782,\\;\\; \\tilde c=\\tanh(0{,}974)=0{,}751,\\;\\; o=\\sigma(0{,}388)=0{,}596",
          "comment": "вход, забывание, кандидат, выход"
        },
        {
          "latex": "c_2=f\\,c_1+i\\,\\tilde c=0{,}782\\cdot 0{,}169+0{,}622\\cdot 0{,}751=0{,}598",
          "comment": "сколько старой памяти оставить + сколько нового записать"
        },
        {
          "latex": "h_2=o\\tanh(c_2)=0{,}596\\cdot\\tanh(0{,}598)=0{,}319",
          "comment": "что ячейка выдаёт на этом шаге"
        }
      ],
      "note": "«good» даёт крупный кандидат c̃. Старая память сохраняется на 78 %, и c вырастает втрое: здесь сеть «запомнила» позитив."
    },
    {
      "title": "LSTM, шаг t = 3: «movie»",
      "code": "z = E[idx] @ W + h * U + b",
      "lines": [
        {
          "latex": "\\begin{aligned}z_{i} &= 0{,}1\\cdot 0{,}5 + 0{,}5\\cdot 0{,}2 + 0{,}319\\cdot 0{,}3 + 0 = 0{,}246\\\\z_{f} &= 0{,}1\\cdot 0{,}3 + 0{,}5\\cdot (-0{,}1) + 0{,}319\\cdot 0{,}2 + 1 = 1{,}044\\\\z_{\\tilde c} &= 0{,}1\\cdot 1 + 0{,}5\\cdot 0{,}3 + 0{,}319\\cdot 0{,}5 + 0 = 0{,}41\\\\z_{o} &= 0{,}1\\cdot 0{,}4 + 0{,}5\\cdot 0{,}1 + 0{,}319\\cdot 0{,}2 + 0 = 0{,}154\\end{aligned}",
          "comment": "вход x3 = (0,1; 0,5), прошлое h = 0,319"
        },
        {
          "latex": "i=\\sigma(0{,}246)=0{,}561,\\;\\; f=\\sigma(1{,}044)=0{,}74,\\;\\; \\tilde c=\\tanh(0{,}41)=0{,}388,\\;\\; o=\\sigma(0{,}154)=0{,}538",
          "comment": "вход, забывание, кандидат, выход"
        },
        {
          "latex": "c_3=f\\,c_2+i\\,\\tilde c=0{,}74\\cdot 0{,}598+0{,}561\\cdot 0{,}388=0{,}66",
          "comment": "сколько старой памяти оставить + сколько нового записать"
        },
        {
          "latex": "h_3=o\\tanh(c_3)=0{,}538\\cdot\\tanh(0{,}66)=0{,}312",
          "comment": "что ячейка выдаёт на этом шаге"
        }
      ],
      "note": "«movie» нейтрально: кандидат слабый, f удерживает 74 % памяти. c даже чуть растёт, h почти не меняется — оценка «good» дожила до конца предложения."
    },
    {
      "title": "То же для «bad movie»: [0, 4, 3]",
      "code": "H.append(lstm(train_pad[1]))",
      "lines": [
        {
          "latex": "\\begin{array}{llll}t=1\\;\\text{PAD}: & \\tilde c=0 & c_1=0 & h_1=0\\\\t=2\\;\\text{bad}: & \\tilde c=-0{,}611 & c_2=-0{,}254 & h_2=-0{,}106\\\\t=3\\;\\text{movie}: & \\tilde c=0{,}194 & c_3=-0{,}081 & h_3=-0{,}042\\end{array}",
          "comment": "те же формулы и те же веса — меняются только строки E"
        }
      ],
      "note": "Шаг PAD: x = (0; 0), поэтому z = b, кандидат tanh(0) = 0 и память остаётся пустой. В настоящем Keras строка PAD случайная и не замаскирована (mask_zero=False), так что нули слева слегка «шумят». «bad» даёт отрицательный кандидат, и h уходит в минус."
    },
    {
      "title": "Dense + sigmoid: состояние → вероятность",
      "code": "model.add(Dense(1, activation=\"sigmoid\"))",
      "lines": [
        {
          "latex": "p_1=\\sigma(w\\,h+b)=\\sigma(2\\cdot 0{,}312+0)=\\sigma(0{,}623)=0{,}6509",
          "comment": "«very good movie», метка y = 1"
        },
        {
          "latex": "p_2=\\sigma(2\\cdot (-0{,}042)+0)=\\sigma(-0{,}083)=0{,}4792",
          "comment": "«bad movie», метка y = 0"
        }
      ],
      "note": "Классификатор видит только последнее h — одно число на всё предложение. В книге это вектор из 32 чисел, и Dense берёт их взвешенную сумму."
    },
    {
      "title": "Ошибка: binary_crossentropy",
      "code": "model.compile(loss=\"binary_crossentropy\", ...)",
      "lines": [
        {
          "latex": "L=-\\frac1N\\sum_k\\big[y_k\\ln p_k+(1-y_k)\\ln(1-p_k)\\big]",
          "comment": "среднее по батчу; batch_size = 2 — оба предложения в одном шаге"
        },
        {
          "latex": "L=-\\tfrac12\\big[\\ln 0{,}6509+\\ln(1-0{,}4792)\\big]=-\\tfrac12\\big[-0{,}4294+(-0{,}6524)\\big]=0{,}5409",
          "comment": "для y = 1 работает только ln p, для y = 0 — только ln(1 − p)"
        }
      ],
      "note": "Сеть, которая всегда отвечает p = 0,5, получила бы L = ln 2 ≈ 0,693. У нас уже меньше: круглые веса подобраны так, что «good» тянет вверх, а «bad» вниз."
    },
    {
      "title": "Градиент: куда двигать веса Dense",
      "code": "g_w = np.mean((p - labels) * H);  g_b = np.mean(p - labels)",
      "lines": [
        {
          "latex": "\\frac{\\partial L}{\\partial w}=\\frac1N\\sum_k (p_k-y_k)\\,h_k,\\qquad \\frac{\\partial L}{\\partial b}=\\frac1N\\sum_k (p_k-y_k)",
          "comment": "производные сигмоиды и log-loss сокращаются в простое «ошибка × вход»"
        },
        {
          "latex": "g_w=\\tfrac12\\big[(-0{,}3491)\\cdot 0{,}312+0{,}4792\\cdot(-0{,}042)\\big]=-0{,}0644",
          "comment": "оба слагаемых отрицательные → w надо увеличить"
        },
        {
          "latex": "g_b=\\tfrac12\\big[(-0{,}3491)+0{,}4792\\big]=0{,}0651",
          "comment": "положительный → b надо уменьшить"
        }
      ],
      "note": "В настоящем fit TensorFlow так же дифференцирует и остальное — W, U, b LSTM и таблицу E — через все шаги t назад (backpropagation through time). Мини-код обновляет только Dense, чтобы формулы остались на виду."
    },
    {
      "title": "RMSprop: шаг обновления",
      "code": "model.compile(optimizer=\"rmsprop\", ...)   # η = 0.001, ρ = 0.9",
      "lines": [
        {
          "latex": "v\\leftarrow\\rho\\,v+(1-\\rho)\\,g^2,\\qquad w\\leftarrow w-\\eta\\,\\frac{g}{\\sqrt v+\\varepsilon}",
          "comment": "v — скользящее среднее квадрата градиента; в начале v = 0"
        },
        {
          "latex": "v_w=0{,}1\\cdot(-0{,}0644)^2=41{,}412\\cdot10^{-5},\\qquad w=2-0{,}001\\cdot\\frac{-0{,}0644}{\\sqrt{v_w}}=2{,}0032",
          "comment": ""
        },
        {
          "latex": "\\frac{g}{\\sqrt{0{,}1\\,g^2}}=\\frac{\\operatorname{sign}(g)}{\\sqrt{0{,}1}}\\approx\\pm3{,}16\\;\\Rightarrow\\;\\Delta w=+0{,}0032,\\;\\Delta b=-0{,}0032",
          "comment": "на первом шаге размер градиента сокращается — оба веса сдвигаются ровно на η/√0,1 ≈ 0,0032"
        }
      ],
      "note": "Шаг RMSprop зависит не от величины градиента, а от его отношения к недавней истории: у весов с редкими маленькими градиентами шаг не тонет, у шумных — не взрывается. Поэтому с η = 0,001 loss падает медленно (см. «Результат запуска»), и книге нужны 30 эпох."
    }
  ]
};
