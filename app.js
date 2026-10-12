// Рендер шпаргалки из data/formulas.js: оглавление по блокам, карточки
// со всеми 7 полями, поиск по названию, копирование LaTeX.

const DEFAULT_BLOCKS = {
  E: "Персептрон и ADALINE",
  A: "Основы обучения",
  B: "Логистическая регрессия",
  C: "Метод опорных векторов",
  D: "Деревья решений и k ближайших соседей",
  F: "Подготовка данных",
  G: "Снижение размерности",
  H: "Оценка качества моделей",
  I: "Ансамблевые методы",
  J: "Анализ текстовых данных",
  K: "Регрессионный анализ",
  L: "Кластеризация",
  M: "Многослойные нейронные сети",
  N: "Функции активации нейронных сетей"
};

const ACTIVE_BOOK_CONFIG = typeof BOOK_CONFIG === "object" ? BOOK_CONFIG : {};
const BLOCKS = ACTIVE_BOOK_CONFIG.blocks || DEFAULT_BLOCKS;
const SECTION_PREFIX = ACTIVE_BOOK_CONFIG.sectionPrefix || "Блок";

function sourceReference(formula, compact = false) {
  if (formula.page !== undefined && formula.page !== null && formula.page !== "") {
    return compact ? " стр. " + formula.page : "(" + formula.chapter + ", стр. " + formula.page + ")";
  }
  return compact ? " " + formula.chapter : "(" + formula.chapter + ")";
}

function tex(el, latex, displayMode) {
  if (typeof katex === "undefined") { el.textContent = latex; return; }
  katex.render(latex, el, { displayMode: !!displayMode, throwOnError: false });
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

const EXTRA_LINK_ALIASES = {
  "unit-step-function": ["функция Хевисайда", "ступенчатая функция"],
  "net-input": ["чистый вход", "взвешенная сумма", "скалярное произведение", "w^T x"],
  "perceptron-rule": ["правило обучения персептрона"],
  "sse-cost": ["SSE", "сумма квадратов ошибок", "сумма квадратов"],
  "batch-gradient-descent": ["градиентный спуск", "градиента", "градиент"],
  "sse-gradient": ["ADALINE", "градиент функции стоимости SSE"],
  "standardization": ["стандартизация", "стандартизованных", "стандартное отклонение"],
  "sgd-update": ["SGD", "обновление весов", "стохастический градиентный спуск"],
  "adaptive-learning-rate": ["адаптивный темп обучения", "темп обучения"],
  "odds-ratio": ["отношение шансов", "шансы"],
  "logit": ["логит", "logit"],
  "sigmoid": ["сигмоида", "сигмоиды", "сигмоидой", "сигмоидальная функция", "логистическая функция", "активация", "активации"],
  "likelihood": ["формула 7", "формулы 7", "правдоподобие", "правдоподобия", "функция правдоподобия"],
  "log-likelihood": ["формула 8", "формулы 8", "логарифмическое правдоподобие", "логарифмического правдоподобия", "лог-правдоподобие"],
  "logistic-cost": ["логистическая функция стоимости", "log-loss", "перекрёстная энтропия", "перекрестная энтропия"],
  "sigmoid-derivative": ["производная сигмоиды", "производной сигмоиды", "производная"],
  "l2-regularization": ["L2", "L2-регуляризация", "L2-регуляризации", "регуляризация"],
  "svm-margin": ["зазор", "маржа"],
  "svm-soft-margin": ["мягкий зазор"],
  "kernel-trick": ["ядерный трюк", "ядерная функция", "ядро", "ядра"],
  "rbf-kernel": ["RBF", "гауссово ядро"],
  "information-gain": ["прирост информации"],
  "entropy": ["энтропия", "энтропии", "энтропийным"],
  "gini-impurity": ["Джини", "джини"],
  "classification-error": ["ошибка классификации"],
  "minkowski-distance": ["расстояние Минковского", "Minkowski"],
  "min-max-scaling": ["минимаксное масштабирование", "масштабирование"],
  "l1-regularization": ["L1", "L1-регуляризация", "L1-регуляризации"],
  "covariance": ["ковариация", "ковариации"],
  "eigen-decomposition": ["собственный вектор", "собственные значения", "собственных значений", "собственные"],
  "explained-variance-ratio": ["доля объяснённой дисперсии", "доля объясненной дисперсии"],
  "pca-projection": ["PCA", "главные компоненты", "главная компонента"],
  "lda-mean-vector": ["вектор средних"],
  "lda-within-scatter": ["внутриклассовая матрица разброса"],
  "lda-between-scatter": ["межклассовая матрица разброса", "LDA"],
  "kernel-matrix": ["матрица ядра"],
  "polynomial-kernel": ["полиномиальное ядро"],
  "sigmoid-kernel": ["сигмоидное ядро"],
  "kernel-centering": ["центрирование матрицы ядра"],
  "error-accuracy": ["верность", "accuracy", "ошибка"],
  "fpr-tpr": ["FPR", "TPR", "ложно-положительных", "истинно-положительных"],
  "precision-recall": ["точность", "полнота", "precision", "recall"],
  "f1-score": ["F1"],
  "majority-voting": ["мажоритарное голосование"],
  "ensemble-error": ["ошибка ансамбля", "ансамбль", "ансамбля"],
  "weighted-majority-voting": ["взвешенное мажоритарное голосование"],
  "probability-voting": ["голосование вероятностями"],
  "adaboost": ["AdaBoost"],
  "tf-idf": ["tf-idf", "tfidf"],
  "multiple-linear-regression": ["множественная линейная регрессия", "линейная регрессия"],
  "pearson-correlation": ["корреляция Пирсона"],
  "mse": ["MSE", "среднеквадратичная ошибка", "средний квадрат ошибки"],
  "r-squared": ["R^2", "коэффициент детерминации"],
  "ridge-regression": ["гребневая регрессия", "ridge"],
  "lasso-regression": ["лассо", "Lasso"],
  "elastic-net": ["эластичная сеть", "Elastic Net"],
  "polynomial-regression": ["полиномиальная регрессия"],
  "squared-euclidean-distance": ["квадратичное евклидово расстояние", "евклидово", "расстояние"],
  "kmeans-sse": ["k-means", "kmeans", "внутрикластерная сумма квадратов"],
  "fcm-objective": ["нечёткое c-means", "нечеткое c-means"],
  "fcm-membership": ["степень принадлежности"],
  "silhouette": ["силуэтный коэффициент", "кластер", "кластера", "кластере"],
  "mlp-hidden-activation": ["нейрон", "нейрона", "нейросети", "активация узла скрытого слоя"],
  "mlp-forward-propagation": ["прямое распространение"],
  "mlp-cost": ["функция стоимости MLP"],
  "backpropagation": ["обратное распространение", "backpropagation", "backprop"],
  "gradient-checking": ["проверка градиента"],
  "softmax": ["softmax", "многоклассовая", "многоклассовой"],
  "tanh-activation": ["tanh", "гиперболический тангенс"],
  "log-likelihood-partial-weight": ["частная производная логарифмического правдоподобия по весу"],
  "sigmoid-chain-weight": ["производная сигмоиды по весу"],
  "logistic-gradient": ["градиент логистической регрессии"]
};

let AUTO_LINK_TARGETS = [];
const anchorReturnStack = [];

function isStartBoundary(text, index) {
  if (index <= 0 || index >= text.length) return true;
  return !/[\p{L}\p{N}_]/u.test(text[index - 1]);
}

function isEndBoundary(text, index) {
  if (index <= 0 || index >= text.length) return true;
  return !/[\p{L}\p{N}_]/u.test(text[index]);
}

function normalizeLinkText(text) {
  return String(text).toLocaleLowerCase("ru-RU");
}

function compactName(name) {
  return name
    .replace(/\s*\([^)]*\)/g, "")
    .replace(/\s+—\s+.*/g, "")
    .trim();
}

function addAlias(targets, formulaId, value, priority) {
  const alias = String(value || "").trim();
  if (alias.length < 2) return;
  targets.push({
    id: formulaId,
    alias,
    normalizedAlias: normalizeLinkText(alias),
    priority
  });
}

function setupAutoLinkTargets(formulas) {
  const targets = [];

  for (const formula of formulas) {
    const generatedAliases = new Set();
    const manualAliases = new Set(EXTRA_LINK_ALIASES[formula.id] || []);
    const shortName = compactName(formula.name);

    generatedAliases.add(formula.name);
    generatedAliases.add(shortName);
    generatedAliases.add(`${formula.order}. ${shortName}`);
    generatedAliases.add(`${formula.order}.${shortName}`);
    generatedAliases.add(`карточка ${formula.order}`);
    generatedAliases.add(`карточки ${formula.order}`);

    for (const alias of generatedAliases) addAlias(targets, formula.id, alias, 0);
    for (const alias of manualAliases) addAlias(targets, formula.id, alias, 1);
  }

  const deduped = [];
  const seen = new Set();
  for (const target of targets.sort((a, b) => b.priority - a.priority)) {
    const key = target.id + "\0" + target.normalizedAlias;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(target);
  }

  AUTO_LINK_TARGETS = deduped.sort((a, b) => b.alias.length - a.alias.length || b.priority - a.priority);
}

function makeTextLink(text, formulaId) {
  const anchor = el("a", "internal-link", text);
  anchor.href = typeof ML_SITE !== "undefined" ? wikiUrl(formulaId) : "#formula-" + formulaId;
  return anchor;
}

function collectAutoLinks(text, currentFormulaId) {
  const matches = [];
  const normalizedText = normalizeLinkText(text);

  for (const target of AUTO_LINK_TARGETS) {
    if (target.id === currentFormulaId) continue;

    let start = normalizedText.indexOf(target.normalizedAlias);
    while (start !== -1) {
      const end = start + target.normalizedAlias.length;
      if (isStartBoundary(text, start) && isEndBoundary(text, end)) {
        matches.push({ start, end, id: target.id });
      }
      start = normalizedText.indexOf(target.normalizedAlias, start + target.normalizedAlias.length);
    }
  }

  return matches;
}

function appendLinkedText(parent, text, currentFormulaId, manualLinks = []) {
  const matches = [];
  for (const link of manualLinks) {
    if (!link || !link.text || !link.formulaId || link.formulaId === currentFormulaId) continue;
    let index = text.indexOf(link.text);
    while (index !== -1) {
      matches.push({ start: index, end: index + link.text.length, id: link.formulaId, manual: true });
      index = text.indexOf(link.text, index + link.text.length);
    }
  }
  matches.push(...collectAutoLinks(text, currentFormulaId));
  matches.sort((a, b) => a.start - b.start || Number(b.manual) - Number(a.manual) || (b.end - b.start) - (a.end - a.start));

  let cursor = 0;
  for (const match of matches) {
    if (match.start < cursor) continue;
    if (cursor < match.start) parent.append(document.createTextNode(text.slice(cursor, match.start)));
    parent.appendChild(makeTextLink(text.slice(match.start, match.end), match.id));
    cursor = match.end;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

const INLINE_MATH_PATTERN = /(?:[A-Za-zΔδφλμηθξε](?:\^\{[^}]+\}|_\{[^}]+\}|_[A-Za-z0-9]+|[A-Za-z0-9\\{}()\s])*:=\s*[^;\n,.—]+|\|[A-Za-z]\|\s*(?:<|>|=|≤|≥|≈|≠)\s*\d+(?:\.\d+)?|\b(?:[JLIpO]|tanh|logistic|sign|argmax|mode)\([^)]*\)|(?:\d+(?:\.\d+)?|[A-Za-zΔδφλμηθξε])\s*(?:<|>|=|≤|≥|≈|≠)\s*(?:\d+(?:\.\d+)?|[A-Za-zΔδφλμηθξε])(?:\s*(?:<|>|=|≤|≥|≈|≠)\s*(?:\d+(?:\.\d+)?|[A-Za-zΔδφλμηθξε]))?|[+\-−]?\d*\.?\d+(?:e[+\-−]?\d+)?[Δδφλμηθξε]|(?<![\p{L}\p{N}_])[ΣΔδφλμηθξε](?![\p{L}\p{N}_])|[^\p{Script=Cyrillic};:,.—«»]+(?:[\\=^_{}∂∇ΣΔδφλμηθξε‖≤≥≈≠∈×±∞₀-₉¹²³⁻½⁽⁾ᵀ][^\p{Script=Cyrillic};:,.—«»]*)+)/gu;

function normalizeInlineLatex(text) {
  return text
    .trim()
    .replace(/−/g, "-")
    // % в LaTeX — начало комментария: без экранирования съедает текст до конца
    .replace(/%/g, "\\%")
    .replace(/½/g, "\\tfrac{1}{2}")
    .replace(/∗/g, "\\ast")
    .replace(/∂/g, "\\partial ")
    .replace(/∇/g, "\\nabla ")
    .replace(/Σ(?=[ᵢⱼₙ_])/g, "\\sum ")
    .replace(/Σ/g, "\\Sigma ")
    .replace(/Δ/g, "\\Delta ")
    .replace(/δ/g, "\\delta ")
    .replace(/φ/g, "\\phi ")
    .replace(/λ/g, "\\lambda ")
    .replace(/η/g, "\\eta ")
    .replace(/θ/g, "\\theta ")
    .replace(/μ/g, "\\mu ")
    .replace(/σ/g, "\\sigma ")
    .replace(/ξ/g, "\\xi ")
    .replace(/ε/g, "\\varepsilon ")
    .replace(/\|([^|]+)\|/g, "\\lvert $1 \\rvert")
    .replace(/≤/g, "\\le")
    .replace(/≥/g, "\\ge")
    .replace(/≠/g, "\\ne")
    .replace(/∈/g, "\\in")
    .replace(/×/g, "\\times")
    .replace(/±/g, "\\pm")
    .replace(/∞/g, "\\infty")
    .replace(/‖([^‖]+)‖₂/g, "\\lVert $1 \\rVert_2")
    .replace(/‖([^‖]+)‖\^2/g, "\\lVert $1 \\rVert^2")
    .replace(/‖([^‖]+)‖/g, "\\lVert $1 \\rVert")
    .replace(/₀/g, "_0")
    .replace(/₁/g, "_1")
    .replace(/₂/g, "_2")
    .replace(/₃/g, "_3")
    .replace(/₄/g, "_4")
    .replace(/₅/g, "_5")
    .replace(/₆/g, "_6")
    .replace(/₇/g, "_7")
    .replace(/₈/g, "_8")
    .replace(/₉/g, "_9")
    .replace(/⁻¹/g, "^{-1}")
    .replace(/¹/g, "^1")
    .replace(/²/g, "^2")
    .replace(/³/g, "^3")
    .replace(/ᵀ/g, "^T")
    .replace(/⁽ⁱ⁾/g, "^{(i)}")
    .replace(/⁽ʲ⁾/g, "^{(j)}")
    .replace(/⁽ˡ⁾/g, "^{(l)}")
    .replace(/ᵢ/g, "_i")
    .replace(/ⱼ/g, "_j")
    .replace(/ₙ/g, "_n")
    // KaTeX запрещает двойной верхний индекс (x^{(i)}^T) — сливаем в один
    .replace(/\^\{([^{}]*)\}\^T/g, "^{$1T}");
}

function collectInlineMath(text) {
  const matches = [];
  for (const match of text.matchAll(INLINE_MATH_PATTERN)) {
    const raw = match[0];
    const start = match.index || 0;
    const trimmedStart = start + raw.search(/\S/);
    const trimmed = raw.trim();
    if (!trimmed || (trimmed.length < 2 && !/^[ΣΔδφλμηθξε]$/.test(trimmed))) continue;
    if (/^[-+*/()[\]{}|\s]+$/.test(trimmed)) continue;
    // snake_case-идентификаторы (X_train_std, inertia_) — это python-код,
    // а не математика с индексами: оставляем обычным текстом
    if (/_[a-z]{2,}|[a-z]_$|\b(?:np|pd|km|sklearn)\./.test(trimmed)) continue;
    matches.push({
      start: trimmedStart,
      end: trimmedStart + trimmed.length,
      latex: normalizeInlineLatex(trimmed)
    });
  }

  const merged = [];
  for (const match of matches) {
    const previous = merged[merged.length - 1];
    const gap = previous ? text.slice(previous.end, match.start).trim() : "";
    const previousUnbalanced = previous &&
      previous.latex.split("{").length > previous.latex.split("}").length;
    if (previous && gap === ":" && match.latex.startsWith("=")) {
      previous.end = match.end;
      previous.latex = previous.latex + " :=" + match.latex.slice(1);
    } else if (previousUnbalanced && gap === ",") {
      // Запятая внутри фигурных скобок (w_{j,k}) — регулярка режет по ней,
      // склеиваем обратно, пока скобки не закроются
      previous.end = match.end;
      previous.latex = previous.latex + "," + match.latex;
    } else if (previous && gap === "" && /^(?:=|\\ne|\\le|\\ge|\\approx|\\in|<|>)/.test(match.latex)) {
      previous.end = match.end;
      previous.latex = previous.latex + " " + match.latex;
    } else {
      merged.push(match);
    }
  }
  return merged;
}

function appendRichText(parent, text, currentFormulaId, manualLinks = []) {
  const value = text || "";
  const mathMatches = collectInlineMath(value);
  let cursor = 0;

  for (const match of mathMatches) {
    if (match.start < cursor) continue;
    if (cursor < match.start) {
      appendLinkedText(parent, value.slice(cursor, match.start), currentFormulaId, manualLinks);
    }
    // Эвристика распознавания математики может ошибаться: рендерим только то,
    // что KaTeX реально парсит, иначе оставляем обычный текст вместо красной ошибки
    const inline = el("span", "inline-math");
    try {
      katex.render(match.latex, inline, { displayMode: false, throwOnError: true });
      parent.appendChild(inline);
    } catch (error) {
      appendLinkedText(parent, value.slice(match.start, match.end), currentFormulaId, manualLinks);
    }
    cursor = match.end;
  }

  if (cursor < value.length) appendLinkedText(parent, value.slice(cursor), currentFormulaId, manualLinks);
}

function linkedEl(tag, className, text, currentFormulaId, manualLinks = []) {
  const node = el(tag, className);
  appendRichText(node, text || "", currentFormulaId, manualLinks);
  return node;
}

function linkedComment(step, currentFormulaId) {
  return linkedEl("p", "derivation-comment", step.comment, currentFormulaId, [step.commentLink]);
}

function scrollToHash(hash) {
  if (!hash) return false;
  const target = document.getElementById(hash.slice(1));
  if (!target) return false;
  target.scrollIntoView({ block: "start" });
  return true;
}

function urlWithHash(hash) {
  const url = new URL(location.href);
  url.hash = hash || "";
  return url.href;
}

function hideAnchorBackButtons() {
  document.querySelectorAll(".anchor-back:not([hidden])").forEach(button => {
    button.hidden = true;
  });
}

function showAnchorBackButton(hash) {
  hideAnchorBackButtons();
  if (!hash) return;
  const target = document.getElementById(hash.slice(1));
  const button = target && target.querySelector(".anchor-back");
  if (button) button.hidden = false;
}

function navigateToAnchor(hash, sourceCard) {
  const returnPoint = {
    hash: sourceCard ? "#" + sourceCard.id : location.hash,
    scrollY: window.scrollY
  };
  if (!scrollToHash(hash)) return;
  anchorReturnStack.push(returnPoint);
  history.pushState({ anchorTarget: hash }, "", urlWithHash(hash));
  showAnchorBackButton(hash);
}

function returnFromAnchor() {
  const returnPoint = anchorReturnStack.pop();
  hideAnchorBackButtons();

  if (!returnPoint) {
    history.back();
    return;
  }

  history.replaceState(history.state, "", urlWithHash(returnPoint.hash));
  window.requestAnimationFrame(() => {
    window.scrollTo({ top: returnPoint.scrollY });
  });

  if (location.hash) showAnchorBackButton(location.hash);
}

function setupAnchorNavigation() {
  if ("scrollRestoration" in history) {
    history.scrollRestoration = "manual";
  }

  document.addEventListener("click", event => {
    const anchor = event.target.closest('a[href^="#formula-"]');
    if (!anchor) return;

    const hash = anchor.getAttribute("href");
    if (!hash || !document.getElementById(hash.slice(1))) return;

    event.preventDefault();
    navigateToAnchor(hash, anchor.closest(".formula-card"));
  });

  window.addEventListener("popstate", () => {
    hideAnchorBackButtons();
    if (anchorReturnStack.length) {
      const returnPoint = anchorReturnStack.pop();
      history.replaceState(history.state, "", urlWithHash(returnPoint.hash));
      window.requestAnimationFrame(() => {
        window.scrollTo({ top: returnPoint.scrollY });
      });
      if (location.hash) showAnchorBackButton(location.hash);
      return;
    }
    if (location.hash && scrollToHash(location.hash)) return;
    window.scrollTo({ top: 0 });
  });
}

function buildCard(f, options = {}) {
  const card = el("article", "formula-card");
  card.id = "formula-" + f.id;
  card.dataset.name = f.name.toLowerCase();

  const anchorBack = el("button", "anchor-back", "← Назад");
  anchorBack.type = "button";
  anchorBack.hidden = true;
  anchorBack.addEventListener("click", returnFromAnchor);
  card.appendChild(anchorBack);

  // Заголовок: номер, название, страница
  const title = el("h3", null, (options.general ? "" : f.order + ". ") + f.name + " ");
  const ref = el("span", "page-ref", options.general ? "" : sourceReference(f));
  title.appendChild(ref);
  card.appendChild(title);

  // Формула + кнопка копирования LaTeX
  const formulaWrap = el("div", "formula-wrap");
  const display = el("div", "formula-display");
  tex(display, f.latex, true);
  formulaWrap.appendChild(display);

  const copyBtn = el("button", "copy-btn", "копировать LaTeX");
  copyBtn.type = "button";
  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(f.latex).then(() => {
      copyBtn.textContent = "скопировано ✓";
      setTimeout(() => { copyBtn.textContent = "копировать LaTeX"; }, 1500);
    });
  });
  formulaWrap.appendChild(copyBtn);
  card.appendChild(formulaWrap);

  // Вывод формулы (только у формул с полем derivation), свёрнут по умолчанию
  if (f.derivation && f.derivation.length) {
    const details = document.createElement("details");
    details.className = "derivation";
    const summary = el("summary", null, "Как выводится");
    details.appendChild(summary);
    const steps = el("ol", "derivation-steps");
    for (const step of f.derivation) {
      const li = el("li");
      const stepTex = el("div", "derivation-latex");
      tex(stepTex, step.latex, true);
      li.appendChild(stepTex);
      li.appendChild(linkedComment(step, f.id));
      steps.appendChild(li);
    }
    details.appendChild(steps);
    card.appendChild(details);
  }

  // Разобранный пример — главная поясняющая часть каждой карточки.
  const example = options.example || WORKED_EXAMPLES[f.id];
  if (example) {
    const details = document.createElement("details");
    details.className = "worked-example";
    details.open = !!options.openExample;
    details.appendChild(el("summary", null, "Разобранный пример"));

    const content = el("div", "worked-example-content");
    content.appendChild(el("h5", null, "1. Реальная задача"));
    content.appendChild(linkedEl("p", "worked-example-question", example.question, f.id));

    content.appendChild(el("h5", null, "2. Что нужно найти"));
    content.appendChild(linkedEl("p", null, example.goal || ("Вычислить значение формулы «" + f.name + "» для данных из задачи."), f.id));

    if (example.hint) {
      content.appendChild(linkedEl("p", "worked-example-hint", example.hint, f.id));
    }

    content.appendChild(el("h5", null, "3. Связываем условие с обозначениями"));
    if (example.values && example.values.length) {
      const values = el("dl", "worked-example-values");
      for (const value of example.values) {
        const dt = el("dt");
        tex(dt, value.symbol, false);
        const dd = el("dd");
        dd.appendChild(el("strong", null, value.value + " — "));
        appendLinkedText(dd, value.meaning, f.id);
        values.append(dt, dd);
      }
      content.appendChild(values);
    } else {
      const values = el("dl", "worked-example-values worked-example-values-general");
      for (const param of f.params) {
        const dt = el("dt");
        tex(dt, param.symbol, false);
        values.append(dt, linkedEl("dd", null, param.desc, f.id));
      }
      content.appendChild(values);
    }

    content.appendChild(el("h5", null, "4. Исходная формула"));
    const sourceFormula = el("div", "worked-example-formula");
    tex(sourceFormula, f.latex, true);
    content.appendChild(sourceFormula);

    content.appendChild(el("h5", null, "5. Подставляем и считаем по шагам"));
    if (example.steps && example.steps.length) {
      const steps = el("ol", "worked-example-steps");
      for (const step of example.steps) {
        const li = el("li");
        const calculation = el("div", "worked-example-calculation");
        tex(calculation, step.latex, true);
        li.appendChild(calculation);
        li.appendChild(linkedEl("p", "worked-example-step-comment", step.comment, f.id));
        steps.appendChild(li);
      }
      content.appendChild(steps);
    } else {
      const calculation = el("div", "worked-example-calculation");
      tex(calculation, example.calculation, true);
      content.appendChild(calculation);
    }

    content.appendChild(el("h5", null, "6. Смысл результата"));
    const answer = el("p", "worked-example-answer");
    appendLinkedText(answer, example.answer, f.id);
    content.appendChild(answer);

    content.appendChild(el("h5", null, "7. Где это применяется"));
    content.appendChild(linkedEl("p", "worked-example-application", example.application || f.examples[0], f.id));
    details.appendChild(content);
    card.appendChild(details);
  }

  if (options.bookView) return card;

  // Год и автор
  const meta = el("p", "discovered");
  meta.append(el("strong", null, "Кто и когда: "));
  appendLinkedText(meta, f.discovered.who + " — " + f.discovered.year + ".", f.id);
  if (f.discovered.note) {
    meta.append(" ");
    meta.append(linkedEl("span", "note", f.discovered.note + ".", f.id));
  }
  card.appendChild(meta);

  // Параметры
  card.appendChild(el("h4", null, "Параметры"));
  const dl = el("dl", "params");
  for (const p of f.params) {
    const dt = el("dt");
    tex(dt, p.symbol, false);
    const dd = linkedEl("dd", null, p.desc, f.id);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
  card.appendChild(dl);

  // Что делает
  card.appendChild(el("h4", null, "Что делает"));
  card.appendChild(linkedEl("p", "description", f.description, f.id));

  // Реальные примеры
  card.appendChild(el("h4", null, "Реальные примеры"));
  const ul = el("ul", "examples");
  for (const ex of f.examples) ul.appendChild(linkedEl("li", null, ex, f.id));
  card.appendChild(ul);

  if (f.resources && f.resources.length) {
    card.appendChild(el("h4", null, "Код и материалы"));
    const resources = el("ul", "resources");
    for (const resource of f.resources) {
      const li = el("li");
      const link = el("a", null, resource.label);
      link.href = resource.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      li.appendChild(link);
      resources.appendChild(li);
    }
    card.appendChild(resources);
  }

  return card;
}

function buildToc(sorted) {
  const toc = el("nav", "toc");
  toc.appendChild(el("h2", null, "Оглавление"));
  let currentBlock = null, list = null;
  for (const f of sorted) {
    if (f.block !== currentBlock) {
      currentBlock = f.block;
      toc.appendChild(el("h3", null, SECTION_PREFIX + " " + currentBlock + ". " + BLOCKS[currentBlock]));
      list = el("ol", "toc-list");
      toc.appendChild(list);
    }
    const li = el("li");
    const a = el("a", null, f.name);
    a.href = "#formula-" + f.id;
    li.appendChild(a);
    li.appendChild(el("span", "page-ref", sourceReference(f, true)));
    list.appendChild(li);
  }
  return toc;
}

function applySearch(query) {
  const q = query.trim().toLowerCase();
  let visible = 0;
  document.querySelectorAll(".formula-card").forEach(card => {
    const match = !q || card.dataset.name.includes(q);
    card.hidden = !match;
    if (match) visible++;
  });
  // Заголовок блока прячем, если в нём не осталось видимых карточек
  document.querySelectorAll(".block-section").forEach(section => {
    section.hidden = !section.querySelector(".formula-card:not([hidden])");
  });
  document.getElementById("empty-state").hidden = visible > 0;
  document.querySelector(".toc").hidden = !!q;
}

document.addEventListener("DOMContentLoaded", () => {
  if (typeof ML_SITE !== "undefined") return;
  const root = document.getElementById("app");
  const sorted = [...FORMULAS].sort((a, b) => a.order - b.order);
  setupAutoLinkTargets(sorted);

  // Поиск
  const searchWrap = el("div", "search-wrap");
  const search = document.createElement("input");
  search.type = "search";
  search.id = "search";
  search.placeholder = "Поиск по названию формулы…";
  search.setAttribute("aria-label", "Поиск по названию формулы");
  search.addEventListener("input", () => applySearch(search.value));
  searchWrap.appendChild(search);
  root.appendChild(searchWrap);

  root.appendChild(buildToc(sorted));

  // Карточки по блокам
  let currentBlock = null, section = null;
  for (const f of sorted) {
    if (f.block !== currentBlock) {
      currentBlock = f.block;
      section = el("section", "block-section");
      section.appendChild(el("h2", "block-title", SECTION_PREFIX + " " + currentBlock + ". " + BLOCKS[currentBlock]));
      root.appendChild(section);
    }
    section.appendChild(buildCard(f));
  }

  const empty = el("p", null, "Ничего не найдено. Попробуйте другое название.");
  empty.id = "empty-state";
  empty.hidden = true;
  root.appendChild(empty);

  setupAnchorNavigation();
  scrollToHash(location.hash);
});
