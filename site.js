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
  if(book.officialExampleIds.length) {
    const official=el('section','official-promo');
    official.append(el('p','eyebrow','Код непосредственно к книге'),el('h2',null,'11 официальных примеров Packt'),el('p',null,'Не наши демонстрации: реальные ноутбуки и скрипты CH01–CH11 из публичного репозитория книги. Для каждого показаны источник, зависимости, объяснение шагов и замечания о совместимости.'));
    official.append(link('Открыть официальный код по главам →',officialIndexUrl(book.id),'primary-link'));
    root.append(official);
  }
  const items=[],groups=[],container=el('div','chapter-list');
  const bookUnits=book.unitIds.map(unitId=>unitById(id,unitId));
  for (const chapter of [...new Set(bookUnits.map(u=>u.chapter))]) {
    const section=el('section','chapter'); section.append(el('h2',null,'Глава '+chapter));
    for(const unit of bookUnits.filter(u=>u.chapter===chapter)) {
      const a=link('',unitUrl(id,unit.id),'reading-entry');a.dataset.searchItem='true';
      a.append(el('span','reading-page',unit.page ? 'Стр. '+unit.page : 'Тема'),el('span',null,unit.lessons.map(l=>l.name).join(' · ')),el('span','arrow','→'));
      section.append(a);items.push({node:a,text:'Глава '+chapter+' '+(unit.page||'')+' '+unit.title+' '+unit.lessons.map(l=>l.name).join(' ')});
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
  actions.append(copy,download);const pre=el('pre');pre.append(el('code',null,example.code));panel.append(actions,pre);root.append(panel);
  const walkthrough=el('section','code-walkthrough');walkthrough.append(el('h2',null,'Разбираем по шагам'));
  const steps=el('ol');example.walkthrough.forEach(step=>steps.append(el('li',null,step)));walkthrough.append(steps);root.append(walkthrough);
  const related=el('section','related-formulas');related.append(el('h2',null,'Математическая основа в Formula Wiki'));
  example.relatedFormulaIds.forEach(id=>{const formula=CATALOG.formulas.find(item=>item.id===canonicalId(id));related.append(link(formula.name+' →',wikiUrl(id),'reference-link'));});root.append(related,officialNav(book,example));
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
