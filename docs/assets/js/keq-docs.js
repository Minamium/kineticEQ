(function () {
  var THEME_KEY = 'jtd-theme';

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
    updateThemeButton(theme);
  }

  function currentLanguage() {
    var path = window.location.pathname || '/';
    return path.indexOf('/en/') >= 0 || /\/en$/.test(path) ? 'en' : 'ja';
  }

  function docsBaseUrl() {
    var meta = document.querySelector('meta[name="keq-doc-baseurl"]');
    var base = meta ? meta.getAttribute('content') || '' : '';
    return base.replace(/\/$/, '');
  }

  function relativeDocPath(path) {
    var base = docsBaseUrl();
    var rel = path || '/';
    if (base && rel === base) {
      rel = '/';
    } else if (base && rel.indexOf(base + '/') === 0) {
      rel = rel.slice(base.length) || '/';
    }
    rel = rel.replace(/\/index\.html$/, '/');
    return rel || '/';
  }

  function withBaseUrl(path) {
    var base = docsBaseUrl();
    var rel = path.charAt(0) === '/' ? path : '/' + path;
    return base + rel;
  }

  function languageTargetPath(targetLang) {
    var rel = relativeDocPath(window.location.pathname || '/');
    if (rel === '/' || rel === '/ja/' || rel === '/en/') {
      return targetLang === 'en' ? '/en/' : '/ja/';
    }
    if (rel.indexOf('/en/') === 0) {
      return targetLang === 'en' ? rel : rel.replace(/^\/en/, '') || '/ja/';
    }
    return targetLang === 'en' ? '/en' + rel : rel;
  }

  function updateLanguageButtons(lang) {
    document.querySelectorAll('[data-keq-lang-button]').forEach(function (button) {
      var targetLang = button.getAttribute('data-keq-lang-button');
      button.classList.toggle('is-active', targetLang === lang);
      if (targetLang === 'ja' || targetLang === 'en') {
        button.setAttribute('href', withBaseUrl(languageTargetPath(targetLang)));
      }
    });
  }

  function shouldHideNavLink(path, lang) {
    var rel = relativeDocPath(path);
    if (rel === '/' || rel === '') return false;
    if (rel === '/ja/' || rel === '/ja' || rel === '/en/' || rel === '/en') return true;
    var isEnglish = rel.indexOf('/en/') === 0;
    return lang === 'en' ? !isEnglish : isEnglish;
  }

  function filterLanguageNav(lang) {
    document.querySelectorAll('.site-nav li, .nav-list li').forEach(function (item) {
      var link = item.querySelector('a[href]');
      if (!link) return;
      var href;
      try {
        href = new URL(link.getAttribute('href'), window.location.href).pathname;
      } catch (err) {
        return;
      }
      item.classList.toggle('keq-nav-hidden', shouldHideNavLink(href, lang));
    });
  }

  function updateThemeButton(theme) {
    var button = document.querySelector('[data-keq-theme-toggle]');
    if (button) button.textContent = theme === 'dark' ? 'Light' : 'Dark';
  }

  document.addEventListener('DOMContentLoaded', function () {
    var saved = window.localStorage.getItem(THEME_KEY);
    var theme = saved === 'dark' || saved === 'light' ? saved : currentTheme();
    applyTheme(theme);

    var lang = currentLanguage();
    document.documentElement.setAttribute('data-keq-lang', lang);
    updateLanguageButtons(lang);
    filterLanguageNav(lang);

    var toggle = document.querySelector('[data-keq-theme-toggle]');
    if (toggle) {
      toggle.addEventListener('click', function () {
        applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
      });
    }
  });
})();
