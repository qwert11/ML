// Генерирует обычные HTML-страницы для GitHub Pages; зависимостей Node нет.
// Исходные данные сохраняются отдельно. На хостинге никакой сборки не требуется.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const books = require('../data/books.cjs');
const python = require('../data/python-examples.cjs');
const officialExamples = require('../data/official-transformer-examples.cjs');
const aliases = { 'attention-softmax': 'softmax', 'batch-gradient-descent': 'gradient-descent', 'attention-output': 'scaled-dot-product-attention' };
const canonical = id => aliases[id] || id;
const formulas = new Map();
const units = [];
const compiledBooks = [];
const codeResults = {};
const checkPython = process.argv.includes('--verify-python');
const cachedPath = path.join(root, 'data/python-results.json');
const previousResults = fs.existsSync(cachedPath) ? JSON.parse(fs.readFileSync(cachedPath, 'utf8')) : {};
const escape = text => String(text).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const generated = [];
function write(file, content) {
  const target = path.resolve(root, file);
  if (!target.startsWith(root + path.sep)) throw Error('Outside repository: ' + file);
  fs.mkdirSync(path.dirname(target), {recursive:true});
  fs.writeFileSync(target, content, 'utf8');
  generated.push(file);
}
for (const book of books) {
  const context = vm.createContext({});
  for (const file of book.files) vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), context);
  const raw = JSON.parse(vm.runInContext('JSON.stringify({formulas:FORMULAS,examples:WORKED_EXAMPLES})', context));
  const bookUnits = new Map();
  for (const entry of raw.formulas.sort((a,b) => a.order-b.order)) {
    const id = canonical(entry.id);
    const chapter = Number(entry.chapter.match(/\d+/)[0]);
    const unitId = entry.page ? `chapter-${chapter}-page-${entry.page}` : `chapter-${chapter}-topic-${entry.block.toLowerCase()}`;
    let unit = bookUnits.get(unitId);
    if (!unit) {
      unit = {id:unitId,bookId:book.id,chapter,page:entry.page || null,title:book.contexts[entry.block][0],context:book.contexts[entry.block][1],lessons:[]};
      bookUnits.set(unitId, unit);
    }
    let formula = formulas.get(id);
    if (!formula) {
      const {page,chapter,order,block,...general} = entry;
      formula = {...general,id,category:book.contexts[entry.block][0],example:raw.examples[entry.id],references:[]};
      formulas.set(id,formula);
    }
    if (!python[entry.id]) throw Error('Missing Python: ' + entry.id);
    const code = python[entry.id];
    const cached = previousResults[entry.id];
    if (checkPython || !cached || cached.code !== code) {
      const result = spawnSync('python', ['-X','utf8','-c',code], {encoding:'utf8', timeout:15000});
      if (result.status !== 0) throw Error(`Python ${entry.id}: ${result.stderr || result.error}`);
      codeResults[entry.id] = {code,output:result.stdout.trim()};
    } else codeResults[entry.id] = cached;
    const lesson = {
      id:entry.id,formulaId:id,name:entry.name,latex:entry.latex,params:entry.params,
      example:raw.examples[entry.id] || {
        question:'Обновим один вес логистической регрессии по двум объектам: текущий вес 0,5, шаг 0,1, метки [1,0], вероятности [0,8;0,3], значения признака [2,1].',
        goal:'Получить новый вес после одного пакетного шага.',
        steps:[{latex:String.raw`\Delta w=0{,}1[(1-0{,}8)2+(0-0{,}3)1]=0{,}01`,comment:'Суммируем поправки двух объектов и умножаем на шаг.'},{latex:'w_{new}=0{,}5+0{,}01=0{,}51',comment:'Добавляем поправку к старому весу.'}],
        answer:'Вес стал 0,51. Первый объект увеличивал вес, второй — уменьшал; суммарная поправка положительна.'
      },
      codeId:entry.id,
      application:entry.examples[0],resources:entry.resources || []
    };
    if (!formula.example) formula.example = lesson.example;
    unit.lessons.push(lesson);
    formula.references.push({bookId:book.id,unitId,lessonId:entry.id,chapter,page:entry.page || null,name:entry.name});
    write(`examples/${entry.id}.py`, '# Авторский учебный пример ML Formula Wiki\n' + code + '\n');
  }
  const list = [...bookUnits.values()];
  list.forEach((unit,index) => {unit.previous = list[index-1]?.id; unit.next = list[index+1]?.id;});
  units.push(...list);
  const {files,contexts,...metadata} = book;
  compiledBooks.push({...metadata,unitIds:list.map(u => u.id),formulaCount:raw.formulas.length});
}
const officialIds = new Set();
for (const example of officialExamples) {
  if (officialIds.has(example.id)) throw Error('Duplicate official example: ' + example.id);
  officialIds.add(example.id);
  if (!compiledBooks.some(book => book.id === example.bookId)) throw Error('Unknown book for official example: ' + example.id);
  for (const formulaId of example.relatedFormulaIds) {
    if (!formulas.has(canonical(formulaId))) throw Error(`Unknown related formula ${formulaId} in ${example.id}`);
  }
  if (checkPython) {
    const syntax = spawnSync('python', ['-X','utf8','-c','import ast,sys; ast.parse(sys.stdin.read())'], {input:example.code,encoding:'utf8',timeout:15000});
    if (syntax.status !== 0) throw Error(`Official Python syntax ${example.id}: ${syntax.stderr || syntax.error}`);
  }
  const attribution = `# Фрагмент официального кода Packt: Mastering Transformers\n# MIT License, Copyright (c) 2021 Packt\n# Источник: ${example.sourceUrl}\n`;
  write(`examples/packt-${example.id}.py`, attribution + example.code + '\n');
  if (example.miniDemo) {
    // Мини-код выполняется построчно под трассировщиком; страница прокручивает шаги.
    // Обычный запуск обязан напечатать ровно то же, что видно в логе трассировки.
    const mini = example.miniDemo, key = 'mini-' + example.id, cached = previousResults[key];
    const traceFile = `data/traces/${key}.json`, code = mini.code;
    if (checkPython || !cached || cached.code !== code || !fs.existsSync(path.join(root, traceFile))) {
      const plain = spawnSync('python', ['-X','utf8','-c',code], {encoding:'utf8',timeout:15000});
      if (plain.status !== 0) throw Error(`Python ${key}: ${plain.stderr || plain.error}`);
      const traced = spawnSync('python', ['-X','utf8',path.join(root,'scripts/trace-python.py')], {input:code,encoding:'utf8',timeout:15000});
      if (traced.status !== 0) throw Error(`Trace ${key}: ${traced.stderr || traced.error}`);
      const trace = JSON.parse(traced.stdout);
      if (trace.output !== plain.stdout.replace(/\r\n/g,'\n')) throw Error(`Trace ${key}: вывод трассировки не совпадает с обычным запуском`);
      write(traceFile, JSON.stringify(trace) + '\n');
      codeResults[key] = {code,output:trace.output.trimEnd()};
    } else codeResults[key] = cached;
    // Число шагов нужно странице сразу: по нему задаётся высота прокрутки до загрузки шагов.
    mini.trace = traceFile;
    mini.steps = JSON.parse(fs.readFileSync(path.join(root, traceFile), 'utf8')).steps.length;
    write(`examples/${key}.py`, code);
  }
}
for (const book of compiledBooks) {
  book.officialExampleIds = officialExamples.filter(example => example.bookId === book.id).map(example => example.id);
}
// Связь между разбором внимания и общей softmax существует в данных, не копируется.
const data = {aliases,formulas:[...formulas.values()],books:compiledBooks,units,python:codeResults,officialExamples};
write('data/python-results.json', JSON.stringify(codeResults,null,2)+'\n');
write('data/catalog.js', '/* Generated by scripts/build-site.cjs */\nconst CATALOG = '+JSON.stringify(data)+';\nconst FORMULAS = CATALOG.formulas;\nconst WORKED_EXAMPLES = Object.fromEntries(FORMULAS.map(f=>[f.id,f.example]));\n');
function page(route,title,description,config) {
  const depth = route.split('/').filter(Boolean).length;
  const base = '../'.repeat(depth) || './';
  const url = 'https://qwert11.github.io/ML/' + route;
  const fallbackLinks = config.kind === 'formula' ? data.formulas.find(f=>f.id===config.id).references.map(r=>`<li><a href="${base}books/${r.bookId}/${r.unitId}/#lesson-${r.lessonId}">${escape(r.bookId)} · глава ${r.chapter}${r.page ? ', стр. '+r.page : ''}</a></li>`).join('') : `<li><a href="${base}Formula-Wiki/">Справочник формул</a></li><li><a href="${base}books/">Учебники</a></li>`;
  write(route+'index.html',`<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(title)} · ML</title><meta name="description" content="${escape(description)}">
<link rel="canonical" href="${url}"><link rel="icon" href="data:,">
<script>const ML_SITE=true;const ML_PAGE=${JSON.stringify(config)};try{document.documentElement.dataset.theme=localStorage.getItem('theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}catch(e){}</script>
<link rel="stylesheet" href="${base}styles.css"><link rel="stylesheet" href="${base}library.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg" crossorigin="anonymous"></script>
<script defer src="${base}data/catalog.js"></script><script defer src="${base}app.js"></script><script defer src="${base}site.js"></script>
</head><body><a class="skip-link" href="#app">К содержимому</a>
<header class="masthead"><a class="brand" href="${base}">ML<span> / учиться понимать</span></a><nav aria-label="Основная навигация"><a href="${base}Formula-Wiki/">Формулы</a><a href="${base}books/">Учебники</a><button id="theme-toggle" type="button" aria-label="Переключить тему">◐</button></nav></header>
<main id="app" class="library-app"><h1>${escape(title)}</h1><p>${escape(description)}</p><noscript><p>Для интерактивных разборов включите JavaScript.</p><ul>${fallbackLinks}</ul></noscript></main>
<footer class="library-footer">ML · Формулы, контекст и практика. Авторские объяснения и учебные примеры.</footer>
</body></html>\n`);
}
page('', 'Математика машинного обучения', 'Читайте учебник по порядку или разбирайте формулу в общей Wiki.', {kind:'home'});
page('Formula-Wiki/', 'Справочник формул', `${data.formulas.length} формулы: обозначения, смысл, расчёты и связи с учебниками.`, {kind:'wiki'});
page('books/', 'Учебники', 'Конспекты по главам и страницам: контекст, формула, пример и Python.', {kind:'books'});
for (const formula of data.formulas) page(`Formula-Wiki/${formula.id}/`,formula.name,formula.description.slice(0,180),{kind:'formula',id:formula.id});
for (const book of compiledBooks) {
  page(`books/${book.id}/`,book.title,book.description,{kind:'book',id:book.id});
  for (const unit of units.filter(u=>u.bookId===book.id)) page(`books/${book.id}/${unit.id}/`,`${book.title} — глава ${unit.chapter}${unit.page ? ', стр. '+unit.page : ': '+unit.title}`,unit.context,{kind:'lesson',bookId:book.id,unitId:unit.id});
  if (book.officialExampleIds.length) {
    page(`books/${book.id}/official-code/`,`Официальные примеры к книге «${book.title}»`,'Код из публичного репозитория Packt с разбором, зависимостями и ссылками на источник.',{kind:'official-code-index',bookId:book.id});
    for (const example of officialExamples.filter(item => item.bookId === book.id)) {
      page(`books/${book.id}/official-code/${example.id}/`,`${example.title} — официальный пример`,example.purpose,{kind:'official-code',bookId:book.id,exampleId:example.id});
    }
  }
}
page('404/', 'Страница не найдена', 'Выберите формулу в справочнике или вернитесь к учебникам.', {kind:'not-found'});
// Pages использует файл 404.html в корне, поэтому его ссылки считаются от /ML/.
write('404.html',fs.readFileSync(path.join(root,'404/index.html'),'utf8').replaceAll('../','/ML/'));
write('.nojekyll','');
write('sitemap.xml','<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+generated.filter(p=>p.endsWith('/index.html')||p==='index.html').map(p=>'<url><loc>https://qwert11.github.io/ML/'+p.replace(/index.html$/,'')+'</loc></url>').join('')+'</urlset>');
console.log(JSON.stringify({formulas:formulas.size,books:compiledBooks.length,readingPages:units.length,python:Object.keys(codeResults).length,officialExamples:officialExamples.length,html:generated.filter(f=>f.endsWith('.html')).length}));
