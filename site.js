/* Общая навигация, Wiki и чтение книг. Все URL существуют как HTML-файлы. */
const SITE_BASE = new URL('.', document.currentScript.src);
const urlFor = route => new URL(route, SITE_BASE).href;
const canonicalId = id => CATALOG.aliases[id] || id;
function wikiUrl(id) { return urlFor('Formula-Wiki/' + canonicalId(id) + '/'); }
const bookById = id => CATALOG.books.find(book => book.id === id);
const unitById = (bookId, id) => CATALOG.units.find(unit => unit.bookId === bookId && unit.id === id);
const officialById = id => CATALOG.officialExamples.find(example => example.id === id);
const bookUrl = id => urlFor('books/' + id + '/');
const unitUrl = (bookId, id) => bookUrl(bookId) + id + '/';
const officialIndexUrl = bookId => bookUrl(bookId) + 'official-code/';
const officialUrl = (bookId, id) => officialIndexUrl(bookId) + id + '/';
const readingLabel = unit => 'Глава ' + unit.chapter + (unit.page ? ' · стр. ' + unit.page : ' · ' + (unit.title || unit.name || 'тема'));
const referenceLabel = reference => readingLabel(unitById(reference.bookId, reference.unitId) || reference);
function link(label, href, className) {
  const a = el('a', className, label); a.href = href; return a;
}
function heading(root, kicker, title, text) {
  const header = el('div', 'page-intro');
  header.append(el('p','eyebrow',kicker),el('h1',null,title));
  if (text) header.append(el('p','lead',text));
  root.append(header);
}
function crumbs(root, entries) {
  const nav = el('nav','breadcrumbs'); nav.setAttribute('aria-label','Путь страницы');
  nav.append(link('ML',urlFor('')));
  for (const [label,href] of entries) { nav.append(el('span',null,'/'),href ? link(label,href) : el('span',null,label)); }
  root.append(nav);
}
function bookCard(book) {
  const card = link('',bookUrl(book.id),'book-card');
  card.append(el('p','book-kicker',book.status),el('h2',null,book.title),el('p','book-authors',book.authors),el('p','book-summary',book.description));
  const meta = el('div','book-meta');
  meta.append(el('span',null,book.formulaCount+' разборов'),el('span',null,book.unitIds.length+' шагов чтения'));
  if(book.officialExampleIds.length) meta.append(el('span',null,book.officialExampleIds.length+' примеров Packt'));
  card.append(meta,el('p','book-open','Читать по порядку →'));
  return card;
}
function showBooks(root, title = true) {
  if (title) {crumbs(root,[['Учебники']]);heading(root,'Книжная полка','Учебники','Контекст книги → формула → пример → Python. Читайте по порядку и переходите к общей математической основе в Wiki.');}
  const grid = el('section','book-grid'); grid.setAttribute('aria-label','Учебники');
  CATALOG.books.forEach(book => grid.append(bookCard(book))); root.append(grid);
}
function showHome(root) {
  heading(root,'Математика • машинное обучение','От формулы к пониманию','Одна база знаний. Два способа учиться: следовать за книгой или разобраться в конкретной формуле.');
  const paths = el('div','study-paths');
  paths.append(link('Исследовать '+CATALOG.formulas.length+' формулы →',urlFor('Formula-Wiki/'),'primary-link'),link('Открыть учебники →',urlFor('books/'),'secondary-link'));
  root.append(paths);
  const flow = el('div','learning-flow');
  flow.append(el('p',null,'Учебник даёт контекст и порядок чтения.'),el('p',null,'Wiki раскрывает математическую основу.'),el('p',null,'Примеры и Python связывают их с практикой.'));
  root.append(flow,el('h2',null,'На книжной полке')); showBooks(root,false);
}
function searchable(root, placeholder, items, groups = []) {
  const wrap = el('div','search-wrap');
  const label = el('label',null,placeholder); label.htmlFor='search';
  const search = el('input'); search.id='search'; search.type='search'; search.placeholder=placeholder;
  const status = el('p','search-status'); status.setAttribute('role','status');
  function filter() {
    const query = search.value.trim().toLocaleLowerCase('ru');
    let count = 0;
    items.forEach(({node,text}) => {node.hidden=!text.toLocaleLowerCase('ru').includes(query); if(!node.hidden)count++;});
    groups.forEach(group => { group.hidden = !group.querySelector('[data-search-item]:not([hidden])'); });
    status.textContent = count ? 'Найдено: '+count : 'Ничего не найдено. Попробуйте другое название или номер страницы.';
  }
  search.addEventListener('input',filter); wrap.append(label,search,status); root.append(wrap); filter();
}
function showWiki(root) {
  crumbs(root,[['Formula Wiki']]);
  heading(root,'Общая база знаний','Справочник формул','Смысл, обозначения, расчёты и Python. У каждой формулы есть ссылки на разборы в учебниках.');
  const items = [], sections = [], container = el('div','formula-directory');
  const categories = [...new Set(CATALOG.formulas.map(f=>f.category))];
  for (const category of categories) {
    const section = el('section','directory-section'); section.append(el('h2',null,category));
    const list = el('div','formula-list');
    for (const formula of CATALOG.formulas.filter(f=>f.category===category)) {
      const a = link('',wikiUrl(formula.id),'formula-entry'); a.dataset.searchItem='true';
      a.append(el('strong',null,formula.name),el('span',null,formula.references.length+' разбор(а) в учебниках →'));
      list.append(a); items.push({node:a,text:formula.name+' '+formula.id+' '+formula.category+' '+formula.description});
    }
    section.append(list); container.append(section); sections.push(section);
  }
  searchable(root,'Найти формулу или тему',items,sections); root.append(container);
}
function pythonPanel(codeId) {
  const sample = CATALOG.python[codeId];
  const section = el('section','python-panel'); section.append(el('h4',null,'Проверяем на Python'));
  section.append(el('p','code-note','Самостоятельный учебный мини-пример. Исходные значения заданы в коде; комментарии объясняют вычисление.'));
  section.append(el('p','code-note',sample.code.includes('import numpy') ? 'Python 3 + NumPy. Установка: python -m pip install numpy' : 'Python 3 · только стандартная библиотека'));
  const actions = el('div','code-actions');
  const copy = el('button',null,'Копировать Python'); copy.type='button';
  copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(sample.code);copy.textContent='Скопировано';}catch{copy.textContent='Выделите код и скопируйте вручную';}});
  const download = link('Скачать .py',urlFor('examples/'+codeId+'.py')); download.download=codeId+'.py';
  actions.append(copy,download);
  const pre = el('pre'); pre.append(el('code',null,sample.code));
  const output = el('pre','python-output'); output.append(el('code',null,sample.output));
  section.append(actions,pre,el('h5',null,'Результат запуска'),output);return section;
}
function showFormula(root,id) {
  const formula = CATALOG.formulas.find(f=>f.id===id);
  if (!formula) return showNotFound(root);
  crumbs(root,[['Formula Wiki',urlFor('Formula-Wiki/')],[formula.name]]);
  heading(root,formula.category,formula.name,'Общая математическая основа и её применение в книгах.');
  const from = new URL(location.href).searchParams.get('from');
  const reference = formula.references.find(r=>r.bookId+'/'+r.unitId+'/'+r.lessonId===from);
  if (reference) root.append(link('← Вернуться к месту чтения: '+referenceLabel(reference),unitUrl(reference.bookId,reference.unitId)+'#lesson-'+reference.lessonId,'return-to-book'));
  const backlinks = el('section','book-references');
  backlinks.append(el('h2',null,'В учебниках'));
  for (const ref of formula.references) backlinks.append(link(bookById(ref.bookId).title+' · '+referenceLabel(ref),unitUrl(ref.bookId,ref.unitId)+'#lesson-'+ref.lessonId,'reference-link'));
  root.append(backlinks);
  const official = CATALOG.officialExamples.filter(example => example.relatedFormulaIds.map(canonicalId).includes(formula.id));
  if (official.length) {
    const sources = el('section','official-references');
    sources.append(el('h2',null,'В официальном коде книги'));
    official.forEach(example => sources.append(link('Глава '+example.chapter+' · '+example.title+' →',officialUrl(example.bookId,example.id),'reference-link')));
    root.append(sources);
  }
  const card = buildCard({...formula,order:0},{general:true,openExample:true});
  root.append(card,pythonPanel(formula.references[0].lessonId));
  if (formula.id==='softmax') {
    const related=el('aside','reading-context');related.append(el('h2',null,'Одна формула — разные задачи'),el('p',null,'В классификации нормализуем оценки классов. В attention нормализуем оценки связи с токенами, отдельно для каждого запроса. Математическая операция одна; ось нормализации и смысл результата различаются.'));
    root.append(related);
  }
}
function showBook(root,id) {
  const book = bookById(id); if(!book)return showNotFound(root);
  crumbs(root,[['Учебники',urlFor('books/')],[book.title]]);
  heading(root,book.authors,book.title,book.description);
  if(book.notice)root.append(el('p','reading-notice',book.notice));
  root.append(el('p','reading-note','Конспект охватывает имеющиеся разборы формул. Страницы без разборов пропущены. Примеры и код написаны для этого справочника.'));
  let last;try{last=localStorage.getItem('reading:'+book.id);}catch{}
  const start = book.unitIds.includes(last) ? last : book.unitIds[0];
  root.append(link(last && start===last ? 'Продолжить чтение →' : 'Начать чтение →',unitUrl(book.id,start),'primary-link'));
  const official=book.officialExampleIds.map(officialById);
  if(official.length) {
    const promo=el('p','official-strip');
    promo.append(el('span',null,'В главах ниже рядом с темами стоит реальный код из репозитория книги (Packt, CH01–CH11). '),link('Все '+official.length+' примеров списком →',officialIndexUrl(book.id)));
    root.append(promo);
  }
  const items=[],groups=[],container=el('div','chapter-list');
  const bookUnits=book.unitIds.map(unitId=>unitById(id,unitId));
  const chapters=[...new Set([...bookUnits.map(u=>u.chapter),...official.map(e=>e.chapter)])].sort((a,b)=>a-b);
  for (const chapter of chapters) {
    const section=el('section','chapter'); section.append(el('h2',null,'Глава '+chapter));
    for(const unit of bookUnits.filter(u=>u.chapter===chapter)) {
      const a=link('',unitUrl(id,unit.id),'reading-entry');a.dataset.searchItem='true';
      a.append(el('span','reading-page',unit.page ? 'Стр. '+unit.page : 'Тема'),el('span',null,unit.lessons.map(l=>l.name).join(' · ')),el('span','arrow','→'));
      section.append(a);items.push({node:a,text:'Глава '+chapter+' '+(unit.page||'')+' '+unit.title+' '+unit.lessons.map(l=>l.name).join(' ')});
    }
    for(const example of official.filter(e=>e.chapter===chapter)) {
      const a=link('',officialUrl(id,example.id),'reading-entry code-entry');a.dataset.searchItem='true';
      const text=el('span');text.append(el('strong',null,example.title));
      const formulasUsed=example.relatedFormulaIds.map(fid=>CATALOG.formulas.find(f=>f.id===canonicalId(fid)).name);
      text.append(el('small',null,(example.miniDemo ? 'Есть разбор на числах · ' : '')+'Формулы: '+formulasUsed.join(', ')));
      a.append(el('span','reading-page','Код Packt'+(example.page ? ' · стр. '+example.page : '')),text,el('span','arrow','→'));
      section.append(a);items.push({node:a,text:'Глава '+chapter+' код packt '+(example.page||'')+' '+example.title+' '+example.purpose+' '+formulasUsed.join(' ')});
    }
    groups.push(section);container.append(section);
  }
  searchable(root,'Найти страницу или тему в книге',items,groups);root.append(container);
}
function readerNav(book,unit) {
  const nav=el('nav','reader-nav');nav.setAttribute('aria-label','Порядок чтения');
  if(unit.previous) {const prev=unitById(book.id,unit.previous);nav.append(link('← '+readingLabel(prev),unitUrl(book.id,prev.id)));}
  else nav.append(link('← К оглавлению',bookUrl(book.id)));
  if(unit.next) {const next=unitById(book.id,unit.next);nav.append(link(readingLabel(next)+' →',unitUrl(book.id,next.id)));}
  else nav.append(link('К оглавлению →',bookUrl(book.id)));
  return nav;
}
function showLesson(root,bookId,unitId) {
  const book=bookById(bookId),unit=unitById(bookId,unitId);if(!book||!unit)return showNotFound(root);
  crumbs(root,[['Учебники',urlFor('books/')],[book.title,bookUrl(book.id)],[readingLabel(unit)]]);
  heading(root,book.title,readingLabel(unit),unit.title);
  const position=book.unitIds.indexOf(unit.id)+1;
  const progress=el('p','reading-progress','Шаг '+position+' из '+book.unitIds.length);root.append(progress,readerNav(book,unit));
  if(book.notice)root.append(el('p','reading-notice',book.notice));
  const context=el('aside','reading-context');context.append(el('h2',null,'Где мы находимся в книге'),el('p',null,unit.context));root.append(context);
  const code=examplesForUnit(unit);
  if(code.length)root.append(crossLinks('Официальный код Packt по теме',code.map(({example,reason})=>['Глава '+example.chapter+(example.page ? ', стр. '+example.page : '')+' · '+example.title+' →',officialUrl(bookId,example.id),reason])));
  const toc=el('nav','lesson-index');toc.setAttribute('aria-label','Разборы на этой странице');
  unit.lessons.forEach(lesson=>toc.append(link(lesson.name,'#lesson-'+lesson.id)));root.append(toc);
  unit.lessons.forEach((lesson,index)=>{
    const formula=CATALOG.formulas.find(f=>f.id===lesson.formulaId);
    const section=el('section','book-lesson');section.id='lesson-'+lesson.id;
    section.append(el('p','eyebrow','Разбор '+(index+1)+' · '+(unit.page ? 'стр. '+unit.page : 'учебный пример по теме')));
    const card=buildCard({...formula,id:lesson.id,name:lesson.name,latex:lesson.latex,params:lesson.params,order:index+1,chapter:'Глава '+unit.chapter,page:unit.page,examples:[lesson.application],derivation:undefined},{example:lesson.example,bookView:true,openExample:true});
    section.append(card);
    const destination=new URL(wikiUrl(lesson.formulaId));destination.searchParams.set('from',bookId+'/'+unitId+'/'+lesson.id);
    section.append(link('Подробнее о формуле в Wiki →',destination.href,'wiki-callout'),pythonPanel(lesson.codeId));
    if(lesson.resources.length){const sources=el('p','code-note','Материалы к теме: ');lesson.resources.forEach(resource=>sources.append(link(resource.label,resource.url),' '));section.append(sources);}
    root.append(section);
  });
  root.append(readerNav(book,unit));try{localStorage.setItem('reading:'+bookId,unitId);}catch{}
}
function officialNav(book, example) {
  const items=book.officialExampleIds.map(officialById),index=items.findIndex(item=>item.id===example.id);
  const nav=el('nav','reader-nav');nav.setAttribute('aria-label','Официальные примеры по главам');
  nav.append(index>0 ? link('← Глава '+items[index-1].chapter,officialUrl(book.id,items[index-1].id)) : link('← Все примеры',officialIndexUrl(book.id)));
  nav.append(index<items.length-1 ? link('Глава '+items[index+1].chapter+' →',officialUrl(book.id,items[index+1].id)) : link('К оглавлению книги →',bookUrl(book.id)));
  return nav;
}
function showOfficialIndex(root,bookId) {
  const book=bookById(bookId);if(!book||!book.officialExampleIds.length)return showNotFound(root);
  crumbs(root,[['Учебники',urlFor('books/')],[book.title,bookUrl(book.id)],['Официальный код Packt']]);
  heading(root,'CH01–CH11 · MIT License','Официальные примеры к книге','Это код из публичного репозитория PacktPublishing/Mastering-Transformers, а не примеры, придуманные для сайта. Мы добавили русское объяснение и ссылки на математическую основу.');
  const license=el('aside','source-banner');
  license.append(el('strong',null,'Источник и лицензия'),el('p',null,'Mastering Transformers · Copyright © 2021 Packt · MIT License. Фрагменты привязаны к commit f088d75, поэтому ссылка всегда ведёт именно на разобранную версию.'));
  const repo=link('Открыть официальный репозиторий ↗','https://github.com/PacktPublishing/Mastering-Transformers');repo.target='_blank';repo.rel='noopener';license.append(repo);root.append(license);
  const list=el('div','official-list');
  book.officialExampleIds.map(officialById).forEach(example=>{
    const card=link('',officialUrl(book.id,example.id),'official-entry');
    card.append(el('span','reading-page','Глава '+example.chapter+(example.page ? ' · стр. '+example.page : '')),el('strong',null,example.title),el('span',null,example.purpose),el('span','arrow','Разобрать код →'));
    list.append(card);
  });
  root.append(list);
}
// Мини-версия по шагам: шаги — кнопками, клавишами и ползунком; страница прокручивается мимо как обычно.
// Шаги записаны при сборке настоящим запуском Python (scripts/trace-python.py).
function pyLine(text) {
  const span=el('span','trace-src'),re=/(#.*$)|(f?"[^"]*"|'[^']*')|\b(import|def|return|for|in)\b|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)/g;
  let last=0,m;
  while((m=re.exec(text))){span.append(text.slice(last,m.index),el('span',m[1]?'tk-com':m[2]?'tk-str':m[3]?'tk-kw':'tk-num',m[0]));last=re.lastIndex;}
  span.append(text.slice(last));return span;
}
// Сокращённый оригинал: код Packt на данных и весах разбора; вывод снят настоящим запуском Keras.
function shortDemo(example) {
  const short=example.miniDemo.keras,key='short-'+example.id,section=el('section','python-panel official-code short-code');section.id='short';
  section.append(el('h2',null,'Сокращённый оригинал: тот же код Keras на двух отзывах'),el('p','short-intro',short.intro));
  const actions=el('div','code-actions'),copy=el('button',null,'Копировать код');copy.type='button';
  copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(short.code);copy.textContent='Скопировано';}catch{copy.textContent='Выделите код ниже вручную';}});
  const download=link('Скачать .py',urlFor('examples/'+key+'.py'));download.download=key+'.py';
  actions.append(copy,download,link('↓ Он же по шагам, без Keras','#mini','mini-toggle'),el('span','code-note',short.run));
  const pre=el('pre'),output=el('pre','python-output');pre.append(el('code',null,short.code));output.append(el('code',null,short.output));
  section.append(actions,pre,el('p','code-note short-checked',short.checked),output);
  return section;
}
// Карта «строка Keras → строки разбора» по пометкам мини-кода: «# ── N. блок ──» и «# Keras: …».
// Пометка действует до следующей пометки или заголовка блока.
function kerasMap(code) {
  const lines=code.replace(/\n$/,'').split('\n'),cuts=[],marks=[];let block='';
  lines.forEach((line,i)=>{
    const head=line.match(/^# ── (\d+\. .+?) ─+$/),mark=line.match(/^\s*# Keras: (.*)$/);
    if(head){block=head[1];cuts.push(i+1);}
    if(mark){marks.push({block,keras:mark[1],from:i+1});cuts.push(i+1);}
  });
  marks.forEach(m=>{let to=(cuts.find(n=>n>m.from)||lines.length+1)-1;while(to>m.from&&!lines[to-1].trim())to--;m.to=to;});
  return marks;
}
function traceDemo(example) {
  const mini=example.miniDemo,key='mini-'+example.id,section=el('section','trace');section.id='mini';
  section.append(el('h2',null,'Разбор на числах: программа по шагам'),el('p','trace-intro',mini.intro));
  const map=el('table','trace-scale trace-map'),mapHead=el('tr');
  ['Блок','Что повторяет из Keras','Где в разборе'].forEach(text=>mapHead.append(el('th',null,text)));map.append(mapHead);
  let lastBlock='';
  kerasMap(mini.code).forEach(m=>{
    const tr=el('tr'),jump=el('button','trace-jump','строки '+m.from+'–'+m.to+' →');jump.type='button';jump.title='Перейти к первому шагу этих строк';
    jump.addEventListener('click',()=>{track.jumpTo?.(m.from,m.to);track.scrollIntoView({behavior:'smooth',block:'start'});});
    const td=el('td');td.append(jump);
    tr.append(el('th',null,m.block===lastBlock ? '' : m.block),el('td','trace-map-keras',m.keras),td);map.append(tr);lastBlock=m.block;
  });
  const actions=el('div','code-actions'),copy=el('button',null,'Копировать код');copy.type='button';
  copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(mini.code);copy.textContent='Скопировано';}catch{copy.textContent='Выделите код ниже вручную';}});
  const download=link('Скачать .py',urlFor('examples/'+key+'.py'));download.download=key+'.py';
  actions.append(copy,download,el('span','code-note','Python 3, без библиотек: python '+key+'.py'));
  const diff=el('details','trace-diff'),scale=el('table','trace-scale'),head=el('tr');
  ['','В книге','Здесь'].forEach(text=>head.append(el('th',null,text)));scale.append(head);
  mini.scale.forEach(row=>{const tr=el('tr');row.forEach((text,i)=>tr.append(el(i?'td':'th',null,text)));scale.append(tr);});
  diff.append(el('summary',null,'Чем отличается от книги'),scale);
  const track=el('div','trace-track'),stage=el('div','trace-stage');
  stage.append(el('p','trace-loading','Загружаю шаги…'));track.append(stage);
  const all=el('details','trace-all'),pre=el('pre'),output=el('pre','python-output');
  pre.append(el('code',null,mini.code));output.append(el('code',null,CATALOG.python[key].output));
  all.append(el('summary',null,'Весь код и полный вывод'),pre,output);
  const mapWrap=el('div','trace-map-wrap');mapWrap.append(el('p','trace-map-title','Где в разборе каждая строка сокращённого оригинала'),map);
  section.append(mapWrap,actions,diff,track,all);
  fetch(urlFor(mini.trace)).then(r=>{if(!r.ok)throw Error(r.status);return r.json();})
    .then(trace=>traceViewer(stage,track,mini.code,trace,key))
    .catch(()=>stage.replaceChildren(el('p','trace-loading','Не удалось загрузить шаги. Код можно скопировать и запустить у себя.')));
  return section;
}
// Кнопки отладчика: иконки в духе VS Code, клавиши те же.
const DEBUG_ICONS={
  restart:'<path d="M12.9 9.2A5 5 0 1 1 11.5 4.4"/><path d="M12.2 1.6v3.2H9"/>',
  continue:'<path d="M3.5 3v10"/><path d="M7 3.2 13 8l-6 4.8z" fill="currentColor"/>',
  over:'<path d="M3 9.5a5 5 0 0 1 10 0"/><path d="M10.6 7.6 13 10l2.2-2.6"/><circle cx="8" cy="13" r="1.6" fill="currentColor" stroke="none"/>',
  into:'<path d="M8 1.5V9"/><path d="M5 6.2l3 3 3-3"/><circle cx="8" cy="13" r="1.6" fill="currentColor" stroke="none"/>',
  out:'<path d="M8 9.5V2"/><path d="M5 4.8l3-3 3 3"/><circle cx="8" cy="13" r="1.6" fill="currentColor" stroke="none"/>',
  back:'<path d="M13 9.5a5 5 0 0 0-10 0"/><path d="M5.4 7.6 3 10 .8 7.4"/><circle cx="8" cy="13" r="1.6" fill="currentColor" stroke="none"/>',
  prev:'<path d="M12.5 8h-9"/><path d="M7 4.5 3.5 8 7 11.5"/>',
  next:'<path d="M3.5 8h9"/><path d="M9 4.5 12.5 8 9 11.5"/>'
};
// По шагам разбора (слово, эпоха) — «Назад»/«Дальше»; по строкам — остальные, как в отладчике.
const DEBUG_BUTTONS=[
  ['restart','Заново','Ctrl+Shift+F5','В начало программы'],
  ['prev','Назад','←','Предыдущий шаг разбора'],
  ['next','Дальше','→','Следующий шаг разбора'],
  ['continue','Продолжить','F5','До следующей точки останова (клик по номеру строки), без них — до конца'],
  ['over','Пройти','F10','Шаг с обходом: вызов функции выполняется целиком'],
  ['into','Зайти','F11','По шагам разбора — этот же шаг по строкам, как в отладчике; по строкам — шаг с заходом внутрь функции, то же — стрелка →'],
  ['out','Выйти','Shift+F11','Доработать текущую функцию и вернуться туда, откуда её вызвали; с верхнего уровня — обратно к шагам разбора'],
  ['back','Назад','←','Строка назад']
];
function traceViewer(stage,track,code,trace,key) {
  const steps=trace.steps,N=steps.length,src=code.replace(/\n$/,'').split('\n'),out=trace.output.replace(/\n$/,'').split('\n');
  // Шаги разбора помечены при сборке (s.k: данные, параметры, слово, выход LSTM, эпоха, итог).
  // Шаг разбора охватывает все строки-шаги после предыдущего до своего включительно и показывает их формулы разом.
  const keys=[];steps.forEach((s,k)=>{if(s.k)keys.push(k);});
  const K=keys.length,keyOf=k=>Math.max(0,keys.findIndex(j=>j>=k)),groupStart=i=>i ? keys[i-1]+1 : 0;
  const modeKey='trace-mode';let coarse=K>1;try{if(localStorage.getItem(modeKey)==='lines')coarse=false;}catch{}
  // Над кодом — строка Keras, которую повторяет текущий шаг: последняя пометка «# Keras: …» выше него.
  let section='';const keras=src.map(line=>{const m=line.match(/^\s*# Keras: (.*)$/);if(m)section=m[1];return section;});
  const shown=(s,names)=>names.map(n=>n+' = '+s.v[n]).join(',  ');
  // Подпись у строки: что она присвоила, что вернула или что напечатала.
  const notes=steps.map((s,k)=>{
    if(s.call)return 'вызов: '+shown(s,s.c);
    if(s.r!==undefined)return 'вернула '+s.r;
    if(s.x)return 'цикл окончен';
    const names=[...new Set([...s.t,...s.c])].filter(n=>n in s.v),before=k?steps[k-1].o:0;
    if(names.length)return shown(s,names);
    return s.o>before ? 'напечатала: '+out.slice(before,s.o).join(' | ').trim() : '';
  });
  const globalsAt=[];steps.forEach((s,k)=>{globalsAt[k]=s.f ? globalsAt[k-1] : s.v;});
  // Глубина вызова на каждом шаге: «Пройти» ищет шаг не глубже текущего, «Выйти» — мельче.
  let depthNow=0;const depth=steps.map(s=>{if(s.call)depthNow++;const d=depthNow;if(s.r!==undefined)depthNow--;return d;});
  const after=(k,test)=>{for(let j=k+1;j<N;j++)if(test(j))return j;return N-1;};
  // Строка → начало оператора, на котором можно остановиться; точки останова помнит браузер.
  const stopAt=new Map();steps.forEach(s=>{for(let n=s.l[0];n<=s.l[1];n++)stopAt.set(n,s.l[0]);});
  const bpKey='trace-bp:'+key;let bps=new Set();
  try{bps=new Set(JSON.parse(localStorage.getItem(bpKey)||'[]').filter(n=>stopAt.get(n)===n));}catch{}
  const target={
    restart:()=>0,
    continue:k=>after(k,j=>bps.has(steps[j].l[0])),
    over:k=>after(k,j=>depth[j]<=depth[k]),
    into:k=>k+1,
    out:k=>after(k,j=>depth[j]<depth[k]),
    back:k=>k-1
  };
  const bar=el('div','trace-bar'),modes=el('div','dbg-modes'),tools=el('div','dbg-tools'),range=el('input'),counter=el('span','trace-counter'),where=el('span','trace-where'),buttons={},modeBtn={};
  [['coarse','По шагам разбора',K],['lines','По строкам',N]].forEach(([name,label,n])=>{
    const b=el('button','dbg dbg-mode');b.type='button';b.title=name==='coarse' ? 'Шаг — слово, эпоха, параметры; формулы шага показаны целиком' : 'Шаг — одна строка программы, как в отладчике';
    b.append(label,el('span','dbg-count',String(n)));
    // Обратно к строкам — на ту строку, где были до перехода к шагам, если она внутри текущего шага.
    b.addEventListener('click',()=>{if(coarse===(name==='coarse'))return;
      setMode(name==='coarse',coarse ? (lineAt>=groupStart(ki)&&lineAt<=keys[ki] ? lineAt : groupStart(ki)) : current);});modeBtn[name]=b;modes.append(b);
  });
  DEBUG_BUTTONS.forEach(([name,label,keys,title])=>{
    const b=el('button','dbg dbg-'+name);b.type='button';b.title=title+' — '+keys;b.setAttribute('aria-label',label+'. '+title);
    b.innerHTML='<svg viewBox="0 0 16 16" aria-hidden="true">'+DEBUG_ICONS[name]+'</svg>';b.append(el('span','dbg-label',label),el('kbd',null,keys));
    b.addEventListener('click',()=>act(name));buttons[name]=b;tools.append(b);
  });
  range.type='range';range.min=0;range.max=N-1;range.step=1;range.setAttribute('aria-label','Шаг выполнения');
  bar.append(modes,tools,range,counter,where);
  const box=(title,cls)=>{const wrap=el('div','trace-box '+cls),body=el('div','trace-body');wrap.append(el('p','trace-label',title),body);return [wrap,body];};
  const [codeBox,codeBody]=box('Код · подсвечены строки этого шага · клик по номеру — точка останова','trace-code');
  const [varsBox,varsBody]=box('Переменные после этого шага','trace-vars');
  const [logBox,logBody]=box('Вывод программы (print)','trace-log');
  const [mathBox,mathBody]=box('Формулы сети · результаты — 6 значащих цифр, как печатает программа; числа внутри формулы — 10','trace-math');
  // Система уравнений текущей части сети: подсвечено уравнение этой строки, ниже — подстановка.
  // У шага разбора подстановок несколько (full): все строки шага по порядку.
  const narrow=matchMedia('(max-width:640px)');
  function renderMath(m) {
    mathBody.replaceChildren();
    if(!m){mathBody.append(el('p','trace-empty','В этой строке готовятся данные — формул сети здесь нет.'));return;}
    const inner=el('div','trace-math-inner');inner.append(el('p','trace-math-title',m.title));
    (m.rows||[]).forEach(row=>{const eq=el('div','trace-eq is-'+row.s);tex(eq,row.t,true);inner.append(eq);});
    (m.full||(m.sub ? [m.sub] : [])).forEach(t=>{const sub=el('div','trace-subst');tex(sub,t,true);inner.append(sub);});
    // Подпись без формул (шаг «Данные») на телефоне не прячется — иначе панель пуста.
    if(m.cap)inner.append(el('p','trace-cap'+(inner.childElementCount>1 ? '' : ' is-only'),m.cap));
    if(m.w){const refs=el('p','trace-cap','Подробно: ');m.w.forEach((id,i)=>{if(i)refs.append(' · ');refs.append(link(CATALOG.formulas.find(f=>f.id===canonicalId(id)).name+' →',wikiUrl(id)));});inner.append(refs);}
    mathBody.append(inner);
    // Длинные формулы ужимаются по ширине панели, но не мельче 75 %: остаток панель прокручивает вбок; по высоте — тоже прокрутка.
    for(let pass=0;pass<3;pass++){const k=Math.min(1,mathBody.clientWidth/inner.scrollWidth);
      if(k>=.995)break;inner.style.fontSize=Math.max(.75,(parseFloat(inner.style.fontSize)||100)/100*k*.98)*100+'%';}
    mathBody.scrollTop=0;
  }
  const rows=src.map((text,i)=>{const row=el('div','trace-line');row.append(el('span','trace-num',String(i+1)),pyLine(text),el('span','trace-val'));codeBody.append(row);return row;});
  const paintBps=()=>rows.forEach((row,i)=>row.firstChild.classList.toggle('is-bp',bps.has(i+1)));
  rows.forEach((row,i)=>{const at=stopAt.get(i+1),num=row.firstChild;if(!at)return;
    num.classList.add('can-bp');num.title='Точка останова: F5 / «Продолжить» остановится здесь';
    num.addEventListener('click',()=>{bps.has(at)?bps.delete(at):bps.add(at);try{localStorage.setItem(bpKey,JSON.stringify([...bps]));}catch{}paintBps();});});
  paintBps();
  const grid=el('div','trace-grid');grid.append(codeBox,mathBox,logBox,varsBox);
  stage.replaceChildren(bar,grid);
  const reveal=(pane,node,center)=>{if(!node)return;const top=node.offsetTop,bottom=top+node.offsetHeight;
    if(center)pane.scrollTop=top-(pane.clientHeight-node.offsetHeight)/2;
    else if(top<pane.scrollTop||bottom>pane.scrollTop+pane.clientHeight)pane.scrollTop=top-8;};
  const group=(title,vars,changed)=>{const g=el('div','trace-group');if(title)g.append(el('p','trace-sub',title));
    const names=Object.keys(vars);if(!names.length)g.append(el('p','trace-empty','пока пусто'));
    names.forEach(name=>{const row=el('div','trace-var'+(changed.includes(name)?' is-changed':''));row.append(el('span','trace-name',name),el('span','trace-value',vars[name]));g.append(row);});
    varsBody.append(g);};
  let current=-1,curFrom=-1,ki=0,lineAt=0;
  // k — строка-шаг, from — первая строка-шаг показываемого шага (по строкам from = k).
  function render(k,from) {
    if(k===current&&from===curFrom)return;current=k;curFrom=from;const s=steps[k];
    // Проигрываем шаги до k: на новой итерации или вызове подписи тела стираются.
    const notesNow=new Map(),callers=[];
    for(let j=0;j<=k;j++){
      const t=steps[j],body=trace.bodies[t.l[0]];
      if(body&&!t.x&&(t.call||t.t.length))for(let n=body[0];n<=body[1];n++)notesNow.delete(n);
      if(t.call)callers.push(t.call);
      if(t.r!==undefined&&j<k)callers.pop();
      notesNow.set(t.l[0],notes[j]);
    }
    const caller=s.f ? callers[callers.length-1] : 0;
    // Строки, выполненные на этом шаге, и переменные, которые они поменяли.
    const cur=new Set(),fresh=new Set(),changed=new Set();
    for(let j=from;j<=k;j++){const t=steps[j];for(let n=t.l[0];n<=t.l[1];n++)cur.add(n);fresh.add(t.l[0]);if(t.f===s.f)t.c.forEach(n=>changed.add(n));}
    rows.forEach((row,i)=>{const n=i+1;
      row.classList.toggle('is-current',cur.has(n));row.classList.toggle('is-caller',n===caller);
      row.lastChild.textContent=notesNow.get(n)||'';row.lastChild.classList.toggle('is-fresh',fresh.has(n));});
    reveal(codeBody,rows[s.l[0]-1],true);
    varsBody.replaceChildren();const ch=[...changed];
    if(s.f){group('внутри '+s.f+'()',s.v,ch);group('глобальные',globalsAt[k],[]);}else group('',s.v,ch);
    reveal(varsBody,varsBody.querySelector('.is-changed'));
    renderMath(coarse&&s.k ? Object.assign({},s.m,s.k) : s.m);
    const before=from ? steps[from-1].o : 0;logBody.replaceChildren();
    if(!s.o)logBody.append(el('p','trace-empty','пока ничего не напечатано'));
    out.slice(0,s.o).forEach((line,i)=>logBody.append(el('div','trace-out'+(i>=before?' is-new':''),line||' ')));
    logBody.scrollTop=logBody.scrollHeight;
    range.value=coarse ? ki : k;counter.textContent='Шаг '+(coarse ? ki+1 : k+1)+' из '+(coarse ? K : N);
    buttons.restart.disabled=coarse ? ki===0 : k===0;buttons.back.disabled=k===0;
    buttons.prev.disabled=ki===0;buttons.next.disabled=ki===K-1;
    buttons.continue.disabled=buttons.over.disabled=k===N-1;buttons.into.disabled=!coarse&&k===N-1;
    buttons.out.disabled=!coarse&&!depth[k]&&K<2;
    where.textContent='Keras: '+(keras[s.l[0]-1]||'—')+(s.f ? '  ·  внутри '+s.f+'(), вызов из строки '+caller : '');
  }
  const goKey=i=>{ki=Math.min(K-1,Math.max(0,i));render(keys[ki],groupStart(ki));};
  const go=k=>{k=Math.min(N-1,Math.max(0,k));ki=keyOf(k);if(coarse)goKey(ki);else render(k,k);};
  function setMode(c,k) {
    coarse=c&&K>1;if(coarse)lineAt=k;try{localStorage.setItem(modeKey,coarse ? 'steps' : 'lines');}catch{}
    bar.classList.toggle('is-coarse',coarse);grid.classList.toggle('is-coarse',coarse);modeBtn.coarse.setAttribute('aria-pressed',String(coarse));modeBtn.lines.setAttribute('aria-pressed',String(!coarse));
    range.max=coarse ? K-1 : N-1;current=-1;go(k);
  }
  // «Зайти» на шаге разбора — его же строки по одной; «Выйти» с верхнего уровня — обратно к шагам разбора.
  const act=name=>{
    if(name==='prev')return goKey(ki-1);
    if(name==='next')return goKey(ki+1);
    if(coarse){
      if(name==='restart')goKey(0);
      else if(name==='into')setMode(false,groupStart(ki));
      else if(name==='over')goKey(ki+1);
      else if(name==='continue')goKey(keyOf(target.continue(keys[ki])));   // шаг разбора со следующей точкой останова
      return;
    }
    if(name==='out'&&!depth[current])return setMode(true,current);
    go(target[name](current));
  };
  // Переход из карты над разбором: первый шаг, выполненный в этих строках.
  track.jumpTo=(from,to)=>{const k=steps.findIndex(s=>s.l[0]>=from&&s.l[0]<=to);if(k>=0)go(k);};
  addEventListener('resize',()=>{const k=current,f=curFrom;current=-1;render(k,f);});
  range.addEventListener('input',()=>coarse ? goKey(Number(range.value)) : go(Number(range.value)));
  // Клавиши отладчика работают, пока экран разбора занимает середину окна.
  // F5 перехватывается и на последнем шаге, иначе браузер перезагрузит страницу.
  addEventListener('keydown',e=>{
    if(e.altKey||e.metaKey||e.target.closest?.('input,textarea,select'))return;
    const rect=track.getBoundingClientRect();if(rect.top>innerHeight/2||rect.bottom<innerHeight/2)return;
    const k=e.key,shift=e.shiftKey;
    const name=e.ctrlKey ? (shift&&k==='F5' ? 'restart' : null)
      : k==='F5'&&!shift ? 'continue' : k==='F10'&&!shift ? 'over' : k==='F11' ? (shift ? 'out' : 'into')
      : k==='ArrowRight'&&!shift ? (coarse ? 'next' : 'into') : k==='ArrowLeft'&&!shift ? (coarse ? 'prev' : 'back') : null;
    if(!name)return;
    e.preventDefault();if(!buttons[name].disabled)act(name);
  });
  setMode(coarse,0);
}
// Темы конспекта, связанные с примером: та же глава или общая формула.
function unitsForExample(example) {
  const ids=new Set(example.relatedFormulaIds.map(canonicalId));
  return CATALOG.units.filter(unit=>unit.bookId===example.bookId).map(unit=>{
    const shared=unit.lessons.filter(lesson=>ids.has(lesson.formulaId)).map(lesson=>lesson.name);
    return {unit,reason:unit.chapter===example.chapter ? 'та же глава' : shared.length ? 'та же формула: '+shared.join(', ') : ''};
  }).filter(item=>item.reason);
}
// Обратная связь: официальный код к теме конспекта.
function examplesForUnit(unit) {
  return CATALOG.officialExamples.filter(example=>example.bookId===unit.bookId).map(example=>{
    const ids=example.relatedFormulaIds.map(canonicalId);
    const shared=unit.lessons.filter(lesson=>ids.includes(lesson.formulaId)).map(lesson=>lesson.name);
    return {example,reason:example.chapter===unit.chapter ? 'код этой главы' : shared.length ? 'использует: '+shared.join(', ') : ''};
  }).filter(item=>item.reason);
}
function crossLinks(title,items) {
  const section=el('section','cross-links');section.append(el('h2',null,title));
  items.forEach(([label,href,reason])=>{const row=el('p','cross-link');row.append(link(label,href));if(reason)row.append(el('span',null,reason));section.append(row);});
  return section;
}
function showOfficialExample(root,bookId,exampleId) {
  const book=bookById(bookId),example=officialById(exampleId);if(!book||!example||example.bookId!==bookId)return showNotFound(root);
  crumbs(root,[['Учебники',urlFor('books/')],[book.title,bookUrl(book.id)],['Официальный код',officialIndexUrl(book.id)],['Глава '+example.chapter]]);
  heading(root,'Официальный пример · глава '+example.chapter+(example.page ? ' · стр. '+example.page : ''),example.title,example.purpose);
  root.append(officialNav(book,example));
  const source=el('aside','source-banner');
  source.append(el('strong',null,'Это пример из официального репозитория книги'),el('p',null,example.sourcePath+' · '+example.cells),el('p','code-note',example.excerptNote));
  if(example.page) source.append(el('p','verified-pages','✓ Сверено с фотографиями русского издания · стр. '+example.page));
  const sourceLink=link('Посмотреть исходную ячейку на GitHub ↗',example.sourceUrl);sourceLink.target='_blank';sourceLink.rel='noopener';source.append(sourceLink);root.append(source);
  const requirements=el('section','example-requirements');requirements.append(el('h2',null,'Что понадобится'));
  const tags=el('div','requirement-tags');example.requires.forEach(item=>tags.append(el('span',null,item)));requirements.append(tags,el('p','compatibility-note',example.compatibility));root.append(requirements);
  const panel=el('section','python-panel official-code');panel.append(el('h2',null,'Код Packt'));
  const actions=el('div','code-actions');
  const copy=el('button',null,'Копировать код');copy.type='button';copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(example.code);copy.textContent='Скопировано';}catch{copy.textContent='Выделите код вручную';}});
  const download=link('Скачать .py',urlFor('examples/packt-'+example.id+'.py'));download.download='packt-'+example.id+'.py';
  actions.append(copy,download);panel.append(actions);
  if(example.miniDemo?.keras)actions.append(link('↓ Сокращённый оригинал','#short','mini-toggle'));
  if(example.miniDemo)actions.append(link('↓ Разобрать на числах по шагам','#mini','mini-toggle'));
  const pre=el('pre');pre.append(el('code',null,example.code));panel.append(pre);root.append(panel);
  if(example.miniDemo?.keras)root.append(shortDemo(example));
  if(example.miniDemo)root.append(traceDemo(example));
  const walkthrough=el('section','code-walkthrough');walkthrough.append(el('h2',null,'Разбираем по шагам'));
  const steps=el('ol');example.walkthrough.forEach(step=>steps.append(el('li',null,step)));walkthrough.append(steps);root.append(walkthrough);
  const related=el('section','related-formulas');related.append(el('h2',null,'Математическая основа в Formula Wiki'));
  example.relatedFormulaIds.forEach(id=>{
    const formula=CATALOG.formulas.find(item=>item.id===canonicalId(id)),row=el('div','formula-ref');
    row.append(link(formula.name+' →',wikiUrl(id),'reference-link'));
    if(formula.references.length){const where=el('p','code-note','В учебниках: ');formula.references.forEach((ref,i)=>{if(i)where.append(' · ');where.append(link(bookById(ref.bookId).authors.split(' · ')[0]+', '+referenceLabel(ref).toLowerCase(),unitUrl(ref.bookId,ref.unitId)+'#lesson-'+ref.lessonId));});row.append(where);}
    related.append(row);
  });
  const units=unitsForExample(example);
  if(units.length)root.append(crossLinks('Где это в конспекте книги',units.map(({unit,reason})=>[readingLabel(unit)+': '+unit.lessons.map(l=>l.name).join(' · ')+' →',unitUrl(unit.bookId,unit.id),reason])));
  root.append(related,officialNav(book,example));
}
function showNotFound(root){heading(root,'404','Страница не найдена','Откройте справочник формул или оглавление нужного учебника.');root.append(link('Справочник формул →',urlFor('Formula-Wiki/'),'primary-link'));showBooks(root,false);}
document.addEventListener('DOMContentLoaded',()=>{
  const legacy=location.hash.match(/^#formula-(.+)$/);
  if(legacy && ML_PAGE.kind==='home' && CATALOG.formulas.some(f=>f.id===canonicalId(legacy[1]))){location.replace(wikiUrl(legacy[1]));return;}
  setupAutoLinkTargets(CATALOG.formulas);
  const root=document.getElementById('app');root.replaceChildren();
  switch(ML_PAGE.kind){
    case 'home':showHome(root);break;
    case 'wiki':showWiki(root);break;
    case 'books':showBooks(root);break;
    case 'book':showBook(root,ML_PAGE.id);break;
    case 'formula':showFormula(root,ML_PAGE.id);break;
    case 'lesson':showLesson(root,ML_PAGE.bookId,ML_PAGE.unitId);break;
    case 'official-code-index':showOfficialIndex(root,ML_PAGE.bookId);break;
    case 'official-code':showOfficialExample(root,ML_PAGE.bookId,ML_PAGE.exampleId);break;
    default:showNotFound(root);
  }
  const toggle=document.getElementById('theme-toggle');
  function refresh(){toggle.setAttribute('aria-label',document.documentElement.dataset.theme==='dark' ? 'Включить светлую тему' : 'Включить тёмную тему');}
  toggle.addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;try{localStorage.setItem('theme',theme);}catch{}refresh();});refresh();
  if(location.hash){const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(target)requestAnimationFrame(()=>target.scrollIntoView());}
});
