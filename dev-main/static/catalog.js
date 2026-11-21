/* Progressive enhancement: links and all specimens work without JavaScript. */
(() => {
  const root = document.getElementById('catalog');
  if (!root) return;
  const cards = Array.from(root.querySelectorAll('.soil-card'));
  const search = document.getElementById('catalog-search');
  const family = document.getElementById('catalog-family');
  const more = document.getElementById('catalog-more');
  const count = document.getElementById('catalog-count');
  const empty = document.getElementById('catalog-empty');
  const normalize = value => value.toLocaleLowerCase('ru').replaceAll('ё', 'е').trim();
  const mobile = matchMedia('(max-width: 1024px)');
  const panel = root.querySelector('.catalog-filter-panel');
  const rail = root.querySelector('.catalog-grid');
  function fitFilters() {
    panel.open = true;
  }
  mobile.addEventListener('change', fitFilters);
  const indexed = cards.map(card => ({ card, text: normalize(card.dataset.search) }));
  const pageSize = () => mobile.matches ? 6 : 12;
  let limit = pageSize();
  function readAddress() {
    const params = new URL(location.href).searchParams;
    search.value = params.get('q') || '';
    const requested = params.get('family') || '';
    family.value = Array.from(family.options).some(option => option.value === requested) ? requested : '';
    const parsed = Number(params.get('limit'));
    limit = Math.min(cards.length, Math.max(pageSize(), Number.isFinite(parsed) ? Math.floor(parsed) : pageSize()));
  }
  readAddress();

  function render(updateAddress = true) {
    const terms = normalize(search.value).split(/\s+/).filter(Boolean);
    const matches = indexed.filter(({ card, text }) => (!family.value || card.dataset.family === family.value)
      && terms.every(term => text.includes(term)));
    cards.forEach(card => { card.hidden = true; });
    matches.slice(0, limit).forEach(({ card }) => { card.hidden = false; });
    count.textContent = `Показано ${Math.min(limit, matches.length)} из ${matches.length} · всего ${cards.length} образцов`;
    empty.hidden = matches.length > 0;
    more.hidden = matches.length <= limit;
    if (updateAddress) {
      const address = new URL(location.href);
      for (const [key, value] of [['q', search.value.trim()], ['family', family.value], ['limit', limit > (mobile.matches ? 6 : 12) ? String(limit) : '']]) {
        if (value) address.searchParams.set(key, value);
        else address.searchParams.delete(key);
      }
      // Filtering also works where browser history writes are unavailable.
      try { history.replaceState(null, '', address); } catch (_) { /* no-op */ }
    }
  }
  search.addEventListener('input', () => { limit = mobile.matches ? 6 : 12; render(); rail.scrollLeft = 0; });
  family.addEventListener('change', () => { limit = mobile.matches ? 6 : 12; render(); rail.scrollLeft = 0; });
  document.getElementById('catalog-reset').addEventListener('click', () => {
    search.value = ''; family.value = ''; limit = mobile.matches ? 6 : 12; render(); rail.scrollLeft = 0; search.focus();
  });
  more.addEventListener('click', () => {
    limit += mobile.matches ? 6 : 12; render();
  });
  panel.hidden = false;
  fitFilters();
  root.querySelector('.catalog-controls').hidden = false;
  render(false);
  window.addEventListener('popstate', () => { readAddress(); render(false); });
  document.querySelectorAll('.sidebar a.nav-item').forEach(link => link.addEventListener('click', () => {
    document.getElementById('sidebar')?.classList.remove('visible');
    document.getElementById('overlay')?.classList.remove('active');
    document.getElementById('burger-menu')?.setAttribute('aria-expanded', 'false');
  }));
})();
