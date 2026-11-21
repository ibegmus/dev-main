/* A single mobile shell; navigation is native anchors when JavaScript is off. */
(() => {
  const mobile = matchMedia('(max-width: 1024px)');
  const root = document.documentElement;
  const chat = document.getElementById('ai-chat');
  const help = document.querySelector('.chat-help');
  const saved = document.getElementById('saved-chats');
  const links = [...document.querySelectorAll('.mobile-section-nav a')];
  const memoryButton = document.getElementById('memory-toggle');
  const chatsButton = document.getElementById('mobile-chats-toggle');

  function updateViewport() {
    const viewport = window.visualViewport;
    root.style.setProperty('--mobile-height', `${Math.round(viewport?.height || innerHeight)}px`);
    root.style.setProperty('--mobile-offset-top', `${Math.round(viewport?.offsetTop || 0)}px`);
    // Focus changes before a touch click. Never move the composer on blur.
    // Only the actual visual viewport shrinking signals a software keyboard.
    const keyboard = viewport && Math.abs(viewport.scale - 1) < .05 && viewport.height < innerHeight * .8;
    root.classList.toggle('mobile-keyboard', mobile.matches && keyboard);
    const shell = document.querySelector('.mobile-shell');
    root.style.setProperty('--mobile-shell-height', `${Math.round(shell.getBoundingClientRect().height)}px`);
  }

  function markNavigation(name) {
    for (const link of document.querySelectorAll('.sidebar a.nav-item[href*="#"]')) {
      const active = (link.hash === '#catalog') === (name === 'catalog');
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }

  function showSection(name, updateAddress = false) {
    markNavigation(name);
    if (!mobile.matches) return;
    root.dataset.mobileView = name;
    for (const link of links) {
      const active = (link.hash === '#catalog') === (name === 'catalog');
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    if (updateAddress) {
      const address = new URL(location.href);
      address.hash = name === 'catalog' ? 'catalog' : 'ai-chat';
      try { history.pushState(null, '', address); } catch (_) { location.hash = address.hash; }
      window.scrollTo({ top: 0, behavior: 'instant' });
    }
  }

  function fit() {
    if (mobile.matches) {
      const search = new URL(location.href).searchParams;
      const requested = location.hash === '#catalog' || (!location.hash && (search.has('q') || search.has('family')));
      showSection(requested ? 'catalog' : 'chat');
      help.open = false;
    } else {
      delete root.dataset.mobileView;
      markNavigation(location.hash === '#ai-chat' ? 'chat' : 'catalog');
      help.open = true;
    }
    updateViewport();
  }

  document.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (mobile.matches && link?.origin === location.origin && link.pathname === location.pathname && ['#catalog', '#ai-chat'].includes(link.hash)) {
      event.preventDefault();
      showSection(link.hash === '#catalog' ? 'catalog' : 'chat', true);
    }
    if (event.target.closest('.saved-chat-open')) {
      saved.classList.remove('is-expanded');
      chatsButton.setAttribute('aria-expanded', 'false');
    }
  });
  memoryButton.addEventListener('click', () => {
    const expanded = chat.classList.toggle('memory-expanded');
    memoryButton.setAttribute('aria-expanded', String(expanded));
  });
  chatsButton.addEventListener('click', () => {
    if (!mobile.matches) return;
    const expanded = saved.classList.toggle('is-expanded');
    chatsButton.setAttribute('aria-expanded', String(expanded));
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.getElementById('sidebar').classList.contains('visible')) {
      document.getElementById('sidebar').classList.remove('visible');
      document.getElementById('overlay').classList.remove('active');
      document.getElementById('burger-menu').setAttribute('aria-expanded', 'false');
      document.getElementById('burger-menu').focus();
    }
  });
  window.addEventListener('hashchange', () => showSection(location.hash === '#catalog' ? 'catalog' : 'chat'));
  window.addEventListener('popstate', () => showSection(location.hash === '#catalog' ? 'catalog' : 'chat'));
  mobile.addEventListener('change', fit);
  window.visualViewport?.addEventListener('resize', updateViewport);
  window.visualViewport?.addEventListener('scroll', updateViewport);
  window.addEventListener('resize', updateViewport);
  document.addEventListener('focusin', updateViewport);
  document.addEventListener('focusout', () => requestAnimationFrame(updateViewport));
  fit();
})();
