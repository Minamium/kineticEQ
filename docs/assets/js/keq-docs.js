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
    document.querySelectorAll('[data-keq-lang-button]').forEach(function (button) {
      button.classList.toggle('is-active', button.getAttribute('data-keq-lang-button') === lang);
    });

    var toggle = document.querySelector('[data-keq-theme-toggle]');
    if (toggle) {
      toggle.addEventListener('click', function () {
        applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
      });
    }
  });
})();
