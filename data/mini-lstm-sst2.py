import math

# Тот же «Сокращённый оригинал», что выше, но без Keras: всё, что
# Keras делает внутри своих функций, здесь написано руками — и прямой
# проход, и обратный (градиенты всех весов), и шаг rmsprop.
# Блоки 1–6 идут в том же порядке и под теми же номерами.
# «# Keras: …» — какую строку Keras повторяет блок под этой пометкой.
# Программа печатает каждое число, которое считает. Все они округлены
# до 6 значащих цифр (функция r в блоке 4), и дальше программа считает
# уже с округлённым числом. Поэтому любое число в выводе получается на
# калькуляторе из напечатанных выше по формуле в строке кода рядом.
# «d» перед именем — производная loss по этой величине (dL/d…).

# ── 1. Данные ──────────────────────────────────────────────
# Keras: sentences = [...]; labels = np.array([1, 0])
# в книге: 67 349 отзывов из SST-2; 1 — хороший, 0 — плохой
sentences = ["very good movie", "bad movie"]
labels = [1, 0]

# ── 2. Словарь: слово → номер ──────────────────────────────
# Keras: words = ["PAD"] + list(...); word2idx = {...}
max_sen_len = max(len(s.split()) for s in sentences)  # самый длинный
# PAD — номер 0: им дополняют короткие отзывы до общей длины
words = ["PAD"] + list(dict.fromkeys(
    [w for s in sentences for w in s.split()]))
word2idx = {w: i for i, w in enumerate(words)}
max_words = max(word2idx.values()) + 1  # слов в словаре = строк в E
print("word2idx:", word2idx)

# ── 3. Отзывы → номера, выравнивание ───────────────────────
# Keras: train = [...]; train_pad = pad_sequences(train, maxlen=...)
train = [[word2idx[w] for w in s.split()] for s in sentences]
# pad_sequences дописывает нули слева: [4, 3] → [0, 4, 3]
train_pad = [[0] * (max_sen_len - len(t)) + t for t in train]
print("train_pad:", train_pad)

# ── 4. Модель: веса слоёв и что каждый слой считает ────────
# Keras: model.add(Embedding(max_words, 2)) + build + get_weights()
# Стартовые веса — те, что напечатал Keras выше («стартовые веса»),
# скопированы оттуда. У Keras они случайные; seed делает их одинаковыми.
# таблица: по строке из 2 чисел на каждое слово словаря
E = [[-0.014649, 0.0398174],     # 0 PAD
     [-0.0321821, -0.0280029],   # 1 very
     [0.00118324, -0.00483059],  # 2 good
     [0.0454806, -0.0449704],    # 3 movie
     [0.0459857, 0.0209465]]     # 4 bad

# Keras: model.add(LSTM(1)) + get_weights() → W, U, b
# одна ячейка, у неё 4 вентиля — по столбцу на вентиль
#         i          f          c~         o
W = [[0.456173, 0.456414, 0.915923, -0.560369],    # от x[0]
     [0.693874, -0.0177481, 0.893304, -0.252504]]  # от x[1]
U = [-0.31614, -0.00752934, 0.302063, -0.899309]   # от h прошлого слова
b = [0.0, 1.0, 0.0, 0.0]  # смещения: у f единица, как делает Keras

# Keras: model.add(Dense(1, activation="sigmoid")) + get_weights()
# один вес на вход h и смещение; _d — чтобы не путать с b ячейки LSTM
w_d, b_d = -0.545483, 0.0


def r(x):
    # Округляет до 6 значащих цифр так, как Python округляет свой
    # 16-значный float. Дальше программа считает уже с округлённым числом,
    # поэтому каждое следующее число сходится из напечатанных. Без этого
    # сеть считала бы с 16 знаками, и последняя напечатанная цифра иногда
    # не сходилась бы с пересчётом. (Keras считает в float32 — ~7 цифр.)
    return float(f"{x:.6g}") + 0.0  # + 0.0: чтобы не печаталось «-0.0»


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Что ячейка LSTM делает с одним словом: вектор слова x и память (h, c)
# после прошлого слова → новая память. Вентили тоже возвращает:
# они понадобятся обратному проходу.
def lstm_step(x, h, c):
    # аргумент каждого вентиля: x·W + h·U + b (столбец W и U на вентиль)
    z = [r(x[0] * W[0][k] + x[1] * W[1][k] + h * U[k] + b[k])
         for k in range(4)]
    print(f"   z_i = {z[0]}  z_f = {z[1]}  z_c = {z[2]}  z_o = {z[3]}")
    i = r(sigmoid(z[0]))        # сколько нового записать
    f = r(sigmoid(z[1]))        # сколько старой памяти оставить
    c_hat = r(math.tanh(z[2]))  # что записать
    o = r(sigmoid(z[3]))        # сколько памяти выдать наружу
    print(f"   i = {i}  f = {f}  c~ = {c_hat}  o = {o}")
    c = r(f * c + i * c_hat)    # новая память
    h = r(o * math.tanh(c))     # выход ячейки
    print(f"   c = {c}  h = {h}")
    return h, c, [i, f, c_hat, o]


# Обратный проход через ячейку для одного слова. На входе dh и dc —
# производные loss по выходу и памяти этого слова (от Dense и от слова
# справа). Наружу — то же для слова слева; вклады в градиенты весов
# копятся в gW, gU, gb, gE.
def lstm_back(idx, hp, cp, i, f, c_hat, o, c, dh, dc):
    x = E[idx]
    tc = math.tanh(c)
    do = r(dh * tc)                      # h = o·tanh(c): по o
    dc = r(dc + dh * o * (1 - tc * tc))  # … и по c, плюс пришедшее справа
    print(f"   do = {do}  dc = {dc}")
    di = r(dc * c_hat)                   # c = f·c_prev + i·c~: по i
    df = r(dc * cp)                      #                      по f
    dc_hat = r(dc * i)                   #                      по c~
    print(f"   di = {di}  df = {df}  dc~ = {dc_hat}")
    # назад через sigmoid и tanh к аргументам вентилей z:
    # производная sigmoid = σ·(1 − σ), производная tanh = 1 − tanh²
    dz = [r(di * i * (1 - i)), r(df * f * (1 - f)),
          r(dc_hat * (1 - c_hat * c_hat)), r(do * o * (1 - o))]
    print(f"   dz_i = {dz[0]}  dz_f = {dz[1]}  dz_c = {dz[2]}  dz_o = {dz[3]}")
    # z = x·W + h_prev·U + b: вклад этого слова в градиенты весов ячейки
    for k in range(4):
        gW[0][k] += x[0] * dz[k]
        gW[1][k] += x[1] * dz[k]
        gU[k] += hp * dz[k]
        gb[k] += dz[k]
    # … и в градиент строки Embedding этого слова: dx = W·dz
    dx = [r(sum(W[j][k] * dz[k] for k in range(4))) for j in range(2)]
    gE[idx][0] += dx[0]
    gE[idx][1] += dx[1]
    # слову слева: dh_prev = U·dz, dc_prev = dc·f
    dh_prev = r(sum(U[k] * dz[k] for k in range(4)))
    dc_prev = r(dc * f)
    print(f"   dx = {dx}  dh_prev = {dh_prev}  dc_prev = {dc_prev}")
    return dh_prev, dc_prev


# ── 5. compile: чем мерить ошибку, как менять веса ─────────
# Keras: model.compile(optimizer="rmsprop", loss="binary_crossentropy")
# loss="binary_crossentropy": средняя ошибка по отзывам
def bce(p):
    return -sum(y * math.log(q) + (1 - y) * math.log(1 - q)
                for y, q in zip(labels, p)) / len(p)


# optimizer="rmsprop": его параметры по умолчанию в Keras
lr, rho, eps = 0.001, 0.9, 1e-7


# шаг rmsprop для одного веса: v копит средний квадрат градиента,
# вес сдвигается на градиент, делённый на корень из v
def rms(theta, v, g):
    v = r(rho * v + (1 - rho) * g ** 2)
    return r(theta - lr * g / math.sqrt(v + eps)), v


def r4(a):  # для сверки с Keras: 4 знака, как печатает он
    return [r4(v) for v in a] if isinstance(a, list) else round(a, 4) + 0.0


# ── 6. fit: обучение ───────────────────────────────────────
# Keras: model.fit(...) — внутри: эпохи
# batch_size=32, а отзывов 2 — оба в одном батче: эпоха = один шаг.
# Эпоха: прямой проход, ошибка, обратный проход, шаг rmsprop.
# v для каждого веса — сначала нули
vE = [[0.0, 0.0] for _ in E]
vW = [[0.0] * 4, [0.0] * 4]
vU = [0.0] * 4
vb = [0.0] * 4
v_wd = v_bd = 0.0
for epoch in range(1, 4):  # epochs=3
    print(f"Epoch {epoch}/3")
    # Keras: model.fit(...) — внутри: прямой проход Embedding → LSTM
    # по каждому отзыву; всё запоминаем — пригодится обратному проходу
    H, mem = [], []
    for k, seq in enumerate(train_pad):
        print(f" отзыв {k + 1} «{sentences[k]}»: вперёд")
        h, c, steps = 0.0, 0.0, []  # перед первым словом памяти нет
        for idx in seq:
            x = E[idx]  # Embedding: номер слова выбирает строку таблицы
            print(f"{words[idx]:>5}: x = {x}  h = {h}  c = {c}")
            h_new, c_new, gates = lstm_step(x, h, c)
            steps.append([idx, h, c] + gates + [c_new, h_new])
            h, c = h_new, c_new
        H.append(h)  # дальше идёт только h после последнего слова
        mem.append(steps)
        print(f"   h_T = {h}  (в Dense пойдёт только оно)")
    # Keras: model.fit(...) — внутри: Dense + sigmoid → вероятность «хороший»
    p = [r(sigmoid(w_d * h + b_d)) for h in H]
    print(f"   p_1 = {p[0]}  p_2 = {p[1]}")
    # metrics=["acc"]: доля отзывов, где p > 0.5 совпало с меткой
    acc = sum((q > 0.5) == y for q, y in zip(p, labels)) / len(p)
    # loss — эту ошибку Keras печатает в строке эпохи
    loss = r(bce(p))
    print(f"   acc = {acc}  loss = {loss}")
    # Keras: model.fit(...) — внутри: обратный проход. Сначала Dense:
    # для пары sigmoid + кросс-энтропия производная по аргументу
    # сигмоиды равна (p − y)/N; по w_d — ещё умножить на вход h_T
    dz_d = [r((q - y) / len(p)) for q, y in zip(p, labels)]
    g_wd = r(sum(d * h for d, h in zip(dz_d, H)))
    g_bd = r(sum(dz_d))
    print(f"   dz_d = {dz_d}  g_wd = {g_wd}  g_bd = {g_bd}")
    # дальше назад через LSTM и Embedding: от последнего слова к первому,
    # вклады всех слов обоих отзывов копятся в одни градиенты
    gE = [[0.0, 0.0] for _ in E]
    gW = [[0.0] * 4, [0.0] * 4]
    gU = [0.0] * 4
    gb = [0.0] * 4
    for k, steps in enumerate(mem):
        dh, dc = r(dz_d[k] * w_d), 0.0  # loss зависит от h_T через Dense
        print(f" отзыв {k + 1}: назад  dh = {dh}  dc = {dc}")
        for idx, hp, cp, i, f, c_hat, o, c, h in reversed(steps):
            print(f"{words[idx]:>5}: назад")
            dh, dc = lstm_back(idx, hp, cp, i, f, c_hat, o, c, dh, dc)
    gE = [[r(v) for v in row] for row in gE]
    gW = [[r(v) for v in row] for row in gW]
    gU = [r(v) for v in gU]
    gb = [r(v) for v in gb]
    print(f"   gE = {gE}")
    print(f"   gW = {gW}  gU = {gU}  gb = {gb}")
    # Keras: model.fit(...) — внутри: шаг rmsprop для каждого веса
    for w in range(max_words):
        for j in range(2):
            E[w][j], vE[w][j] = rms(E[w][j], vE[w][j], gE[w][j])
    for j in range(2):
        for k in range(4):
            W[j][k], vW[j][k] = rms(W[j][k], vW[j][k], gW[j][k])
    for k in range(4):
        U[k], vU[k] = rms(U[k], vU[k], gU[k])
        b[k], vb[k] = rms(b[k], vb[k], gb[k])
    w_d, v_wd = rms(w_d, v_wd, g_wd)
    b_d, v_bd = rms(b_d, v_bd, g_bd)
    print(f"   vE = {vE}  E = {E}")
    print(f"   vW = {vW}  W = {W}")
    print(f"   vU = {vU}  U = {U}  vb = {vb}  b = {b}")
    print(f"   v_wd = {v_wd}  w_d = {w_d}  v_bd = {v_bd}  b_d = {b_d}")
    # так печатает Keras: батч 1 из 1, acc и loss эпохи — посчитаны до шага
    print(f"1/1 - acc: {acc:.4f} - loss: {loss:.4f}")

# Keras: get_weights() после обучения — сверить с разбором (4 знака)
print("после обучения:")
print("E =", r4(E))
print("W =", r4(W))
print("U =", r4(U))
print("b =", r4(b))
print("w_d =", r4(w_d))
print("b_d =", r4(b_d))
