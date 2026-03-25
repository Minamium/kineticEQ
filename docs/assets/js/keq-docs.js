(function () {
  function meta(name, fallback) {
    var node = document.querySelector('meta[name="' + name + '"]');
    return node ? node.getAttribute('content') : fallback;
  }

  function normalizeBaseurl(value) {
    if (!value || value === '/') return '';
    return value.endsWith('/') ? value.slice(0, -1) : value;
  }

  function normalizeRelativePath(path) {
    if (!path) return '/';
    var out = path;
    if (!out.startsWith('/')) out = '/' + out;
    if (out.endsWith('/index.html')) out = out.slice(0, -10);
    if (out === '') return '/';
    return out;
  }

  function joinUrl(baseurl, relativePath) {
    var base = normalizeBaseurl(baseurl);
    var rel = relativePath || '/';
    if (!rel.startsWith('/')) rel = '/' + rel;
    return (base + rel).replace(/\/+/g, '/');
  }

  function currentTheme() {
    var theme = null;
    if (window.jtd && typeof window.jtd.getTheme === 'function') {
      try {
        theme = window.jtd.getTheme();
      } catch (err) {
        theme = null;
      }
    }
    theme = theme || document.documentElement.getAttribute('data-color-scheme');
    theme = theme || document.documentElement.getAttribute('data-theme');
    theme = theme || window.localStorage.getItem('jtd-theme');
    theme = theme || window.localStorage.getItem('color_scheme');
    theme = theme || window.localStorage.getItem('keq-theme');
    if (theme !== 'dark' && theme !== 'light') {
      theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return theme;
  }

  function persistTheme(theme) {
    document.documentElement.setAttribute('data-color-scheme', theme);
    document.documentElement.setAttribute('data-theme', theme);
    try {
      window.localStorage.setItem('jtd-theme', theme);
      window.localStorage.setItem('color_scheme', theme);
      window.localStorage.setItem('keq-theme', theme);
    } catch (err) {
      /* no-op */
    }
  }

  function applyTheme(theme) {
    persistTheme(theme);
    if (window.jtd && typeof window.jtd.setTheme === 'function') {
      try {
        window.jtd.setTheme(theme);
      } catch (err) {
        /* no-op */
      }
    }
    updateActiveButtons(theme);
  }

  function updateActiveButtons(theme) {
    var lang = document.documentElement.getAttribute('data-keq-lang') || 'ja';
    document.querySelectorAll('[data-keq-theme]').forEach(function (button) {
      button.classList.toggle('is-active', button.getAttribute('data-keq-theme') === theme);
    });
    document.querySelectorAll('[data-keq-lang-button]').forEach(function (button) {
      button.classList.toggle('is-active', button.getAttribute('data-keq-lang-button') === lang);
    });
  }

  function languageInfo(relativePath) {
    var normalized = normalizeRelativePath(relativePath);
    var isEnglish = normalized === '/en' || normalized === '/en/' || normalized.startsWith('/en/');
    var alt;
    if (isEnglish) {
      alt = normalized.replace(/^\/en(?=\/|$)/, '');
      if (!alt) alt = '/';
    } else {
      alt = normalized === '/' ? '/en/' : '/en' + normalized;
    }
    return {
      current: isEnglish ? 'en' : 'ja',
      alternate: alt,
    };
  }

  function createButton(label, attrs, onClick) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'keq-page-controls__button';
    button.textContent = label;
    Object.keys(attrs).forEach(function (key) {
      button.setAttribute(key, attrs[key]);
    });
    button.addEventListener('click', onClick);
    return button;
  }

  function injectControls() {
    if (document.querySelector('.keq-page-controls')) return;

    var baseurl = normalizeBaseurl(meta('keq-doc-baseurl', ''));
    var path = window.location.pathname || '/';
    var relativePath = path.startsWith(baseurl) ? path.slice(baseurl.length) || '/' : path;
    var langInfoState = languageInfo(relativePath);
    document.documentElement.setAttribute('data-keq-lang', langInfoState.current);

    var container = document.createElement('div');
    container.className = 'keq-page-controls';

    var langGroup = document.createElement('div');
    langGroup.className = 'keq-page-controls__group';
    var langLabel = document.createElement('span');
    langLabel.className = 'keq-page-controls__label';
    langLabel.textContent = 'Language';
    langGroup.appendChild(langLabel);
    langGroup.appendChild(createButton('日本語', {'data-keq-lang-button': 'ja'}, function () {
      if (langInfoState.current === 'ja') return;
      window.location.assign(joinUrl(baseurl, langInfoState.alternate) + window.location.search + window.location.hash);
    }));
    langGroup.appendChild(createButton('English', {'data-keq-lang-button': 'en'}, function () {
      if (langInfoState.current === 'en') return;
      window.location.assign(joinUrl(baseurl, langInfoState.alternate) + window.location.search + window.location.hash);
    }));

    var themeGroup = document.createElement('div');
    themeGroup.className = 'keq-page-controls__group';
    var themeLabel = document.createElement('span');
    themeLabel.className = 'keq-page-controls__label';
    themeLabel.textContent = 'Theme';
    themeGroup.appendChild(themeLabel);
    themeGroup.appendChild(createButton('Light', {'data-keq-theme': 'light'}, function () {
      applyTheme('light');
    }));
    themeGroup.appendChild(createButton('Dark', {'data-keq-theme': 'dark'}, function () {
      applyTheme('dark');
    }));

    container.appendChild(langGroup);
    container.appendChild(themeGroup);

    var main = document.querySelector('main') || document.querySelector('.main-content') || document.body;
    var heading = main.querySelector('h1');
    if (heading && heading.parentNode) {
      heading.parentNode.insertBefore(container, heading);
    } else {
      main.insertBefore(container, main.firstChild);
    }

    updateActiveButtons(currentTheme());
  }

  document.addEventListener('DOMContentLoaded', function () {
    injectControls();
    updateActiveButtons(currentTheme());
  });
})();
