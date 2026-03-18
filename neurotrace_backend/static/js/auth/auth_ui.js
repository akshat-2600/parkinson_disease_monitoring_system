/* ============================================================
   js/auth/auth_ui.js
   Renders the login and signup screens,
   wires form events, and routes to the correct dashboard.
   ============================================================ */

const AuthUI = (() => {

  /* ── Show / hide auth shell vs main app ───────────────── */
  function showLogin() {
    _renderShell('login');
  }

  function showSignup() {
    _renderShell('signup');
  }

  function hideAuth() {
    const shell = document.getElementById('auth-shell');
    if (shell) shell.remove();
    document.getElementById('sidebar-root').style.display = '';
    document.getElementById('topbar-root').style.display  = '';
    document.getElementById('content-root').style.display = '';
  }

  /* ── Shell injector ────────────────────────────────────── */
  function _renderShell(view) {
    // Hide main app
    document.getElementById('sidebar-root').style.display = 'none';
    document.getElementById('topbar-root').style.display  = 'none';
    document.getElementById('content-root').style.display = 'none';

    // Remove existing shell if any
    document.getElementById('auth-shell')?.remove();

    const shell = document.createElement('div');
    shell.id    = 'auth-shell';
    shell.className = 'auth-shell';
    shell.innerHTML = view === 'login' ? _loginHTML() : _signupHTML();
    document.body.appendChild(shell);

    if (view === 'login')  _bindLogin();
    else                   _bindSignup();
  }

  /* ── Login HTML ────────────────────────────────────────── */
  function _loginHTML() {
    return `
      <div class="auth-card">
        <div class="auth-logo">
          <div class="auth-logo-icon">🧠</div>
          <div>
            <div class="auth-logo-name">NeuroTrace</div>
            <div class="auth-logo-tagline">Parkinson's Intelligence Platform</div>
          </div>
        </div>

        <div class="auth-title">Welcome back</div>
        <div class="auth-subtitle">Sign in to access your dashboard</div>

        <div class="auth-alert" id="loginAlert"></div>

        <form class="auth-form" id="loginForm" autocomplete="on">
          <div class="form-group">
            <label class="form-label">Email address</label>
            <div class="form-input-wrap">
              <svg class="form-input-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
              <input type="email" class="form-input" id="loginEmail"
                     placeholder="doctor@hospital.com" autocomplete="email" required />
            </div>
          </div>

          <div class="form-group">
            <label class="form-label">Password</label>
            <div class="form-input-wrap">
              <svg class="form-input-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              <input type="password" class="form-input" id="loginPassword"
                     placeholder="••••••••" autocomplete="current-password" required />
              <button type="button" class="form-input-toggle" id="loginPassToggle"
                      aria-label="Toggle password visibility">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
              </button>
            </div>
          </div>

          <button type="submit" class="auth-submit" id="loginBtn">
            Sign In
          </button>
        </form>

        <div class="auth-divider">or</div>

        <div style="display:flex;flex-direction:column;gap:8px">
          <div style="font-size:11px;font-family:var(--font-mono);color:var(--text-3);text-align:center;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">Demo credentials</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <button class="btn btn-ghost" onclick="AuthUI.fillDemo('doctor')" style="font-size:12px;justify-content:center">
              👨‍⚕️ Doctor Demo
            </button>
            <button class="btn btn-ghost" onclick="AuthUI.fillDemo('patient')" style="font-size:12px;justify-content:center">
              🧑 Patient Demo
            </button>
          </div>
        </div>

        <div class="auth-footer">
          Don't have an account?
          <button class="auth-link" onclick="AuthUI.showSignup()">Create account</button>
        </div>
      </div>`;
  }

  /* ── Signup HTML ───────────────────────────────────────── */
  function _signupHTML() {
    return `
      <div class="auth-card">
        <div class="auth-logo">
          <div class="auth-logo-icon">🧠</div>
          <div>
            <div class="auth-logo-name">NeuroTrace</div>
            <div class="auth-logo-tagline">Parkinson's Intelligence Platform</div>
          </div>
        </div>

        <div class="auth-steps">
          <div class="auth-step active"  id="step1"></div>
          <div class="auth-step"         id="step2"></div>
        </div>

        <div id="signupStep1">
          <div class="auth-title">Create account</div>
          <div class="auth-subtitle">Choose your role to get started</div>

          <div class="auth-alert" id="signupAlert"></div>

          <form class="auth-form" id="signupForm1">
            <!-- Role -->
            <div class="form-group">
              <label class="form-label">I am a</label>
              <div class="role-selector">
                <label class="role-option">
                  <input type="radio" name="role" value="doctor" id="roleDoctor" />
                  <div class="role-card">
                    <div class="role-icon">👨‍⚕️</div>
                    <div class="role-name">Doctor</div>
                    <div class="role-desc">View all patients</div>
                  </div>
                </label>
                <label class="role-option">
                  <input type="radio" name="role" value="patient" id="rolePatient" />
                  <div class="role-card">
                    <div class="role-icon">🧑</div>
                    <div class="role-name">Patient</div>
                    <div class="role-desc">View my own data</div>
                  </div>
                </label>
              </div>
            </div>

            <!-- Name row -->
            <div class="form-row">
              <div class="form-group">
                <label class="form-label">First name</label>
                <input type="text" class="form-input" id="signupFirst" placeholder="James" required />
              </div>
              <div class="form-group">
                <label class="form-label">Last name</label>
                <input type="text" class="form-input" id="signupLast" placeholder="Harrington" required />
              </div>
            </div>

            <!-- Email -->
            <div class="form-group">
              <label class="form-label">Email address</label>
              <div class="form-input-wrap">
                <svg class="form-input-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
                <input type="email" class="form-input" id="signupEmail" placeholder="you@hospital.com" required />
              </div>
            </div>

            <!-- Password -->
            <div class="form-group">
              <label class="form-label">Password</label>
              <div class="form-input-wrap">
                <svg class="form-input-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                <input type="password" class="form-input" id="signupPassword" placeholder="Min 8 characters" required minlength="8" />
                <button type="button" class="form-input-toggle" id="signupPassToggle" aria-label="Toggle">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                </button>
              </div>
            </div>

            <button type="submit" class="auth-submit" id="signupNext">
              Continue →
            </button>
          </form>
        </div>

        <!-- Step 2: role-specific extras -->
        <div id="signupStep2" style="display:none">
          <div class="auth-title" id="step2Title">Almost there</div>
          <div class="auth-subtitle" id="step2Subtitle">A few more details</div>
          <div class="auth-alert" id="signupAlert2"></div>
          <form class="auth-form" id="signupForm2">
            <!-- Doctor fields -->
            <div id="doctorFields" style="display:none">
              <div class="form-group">
                <label class="form-label">Specialisation</label>
                <input type="text" class="form-input" id="doctorSpec" placeholder="Neurology" />
              </div>
              <div class="form-group">
                <label class="form-label">License Number</label>
                <input type="text" class="form-input" id="doctorLicense" placeholder="NRL-2024-001" />
              </div>
            </div>
            <!-- Patient fields -->
            <div id="patientFields" style="display:none">
              <div class="form-row">
                <div class="form-group">
                  <label class="form-label">Age</label>
                  <input type="number" class="form-input" id="patientAge" placeholder="65" min="1" max="120" />
                </div>
                <div class="form-group">
                  <label class="form-label">Gender</label>
                  <select class="form-input" id="patientGender">
                    <option value="">Select</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
              <div class="form-group">
                <label class="form-label">Diagnosis Year (onset)</label>
                <input type="number" class="form-input" id="patientOnset" placeholder="2020" min="1900" max="2099" />
              </div>
            </div>

            <div style="display:flex;gap:10px;margin-top:4px">
              <button type="button" class="btn btn-ghost" onclick="AuthUI._backToStep1()" style="flex:0 0 auto">
                ← Back
              </button>
              <button type="submit" class="auth-submit" id="signupSubmit" style="flex:1">
                Create Account
              </button>
            </div>
          </form>
        </div>

        <div class="auth-footer">
          Already have an account?
          <button class="auth-link" onclick="AuthUI.showLogin()">Sign in</button>
        </div>
      </div>`;
  }

  /* ── Login binding ─────────────────────────────────────── */
  function _bindLogin() {
    // Password toggle
    document.getElementById('loginPassToggle')?.addEventListener('click', () => {
      const inp = document.getElementById('loginPassword');
      inp.type  = inp.type === 'password' ? 'text' : 'password';
    });

    document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const email    = document.getElementById('loginEmail').value.trim();
      const password = document.getElementById('loginPassword').value;
      const btn      = document.getElementById('loginBtn');

      btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px"></div> Signing in…';
      btn.disabled  = true;

      try {
        const data = await Auth.login(email, password);
        _alert('loginAlert', '', false);  // clear
        _onLoginSuccess(data);
      } catch (err) {
        _alert('loginAlert', err.message, true);
        btn.innerHTML = 'Sign In';
        btn.disabled  = false;
      }
    });
  }

  /* ── Signup binding ────────────────────────────────────── */
  function _bindSignup() {
    // Password toggle
    document.getElementById('signupPassToggle')?.addEventListener('click', () => {
      const inp = document.getElementById('signupPassword');
      inp.type  = inp.type === 'password' ? 'text' : 'password';
    });

    // Step 1 → Step 2
    document.getElementById('signupForm1')?.addEventListener('submit', (e) => {
      e.preventDefault();
      const role = document.querySelector('input[name="role"]:checked')?.value;
      if (!role) { _alert('signupAlert', 'Please select a role', true); return; }
      _goToStep2(role);
    });

    // Step 2 final submit
    document.getElementById('signupForm2')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('signupSubmit');
      btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px"></div> Creating…';
      btn.disabled  = true;

      try {
        const payload = _buildSignupPayload();
        const data    = await Auth.signup(payload);
        _onLoginSuccess(data);
      } catch (err) {
        _alert('signupAlert2', err.message, true);
        btn.innerHTML = 'Create Account';
        btn.disabled  = false;
      }
    });
  }

  function _goToStep2(role) {
    document.getElementById('signupStep1').style.display = 'none';
    document.getElementById('signupStep2').style.display = '';
    document.getElementById('step1').classList.remove('active'); document.getElementById('step1').classList.add('done');
    document.getElementById('step2').classList.add('active');

    if (role === 'doctor') {
      document.getElementById('step2Title').textContent    = 'Doctor details';
      document.getElementById('step2Subtitle').textContent = 'Add your professional info (optional)';
      document.getElementById('doctorFields').style.display = '';
      document.getElementById('patientFields').style.display = 'none';
    } else {
      document.getElementById('step2Title').textContent    = 'Patient profile';
      document.getElementById('step2Subtitle').textContent = 'Help us personalise your dashboard';
      document.getElementById('doctorFields').style.display  = 'none';
      document.getElementById('patientFields').style.display = '';
    }
  }

  function _backToStep1() {
    document.getElementById('signupStep1').style.display = '';
    document.getElementById('signupStep2').style.display = 'none';
    document.getElementById('step1').classList.add('active'); document.getElementById('step1').classList.remove('done');
    document.getElementById('step2').classList.remove('active');
  }

  function _buildSignupPayload() {
    const role = document.querySelector('input[name="role"]:checked')?.value;
    const payload = {
      first_name: document.getElementById('signupFirst').value.trim(),
      last_name:  document.getElementById('signupLast').value.trim(),
      email:      document.getElementById('signupEmail').value.trim(),
      password:   document.getElementById('signupPassword').value,
      role,
    };
    if (role === 'doctor') {
      payload.specialisation  = document.getElementById('doctorSpec').value.trim();
      payload.license_number  = document.getElementById('doctorLicense').value.trim();
    } else {
      payload.age        = parseInt(document.getElementById('patientAge').value)  || undefined;
      payload.gender     = document.getElementById('patientGender').value || undefined;
      payload.onset_year = parseInt(document.getElementById('patientOnset').value) || undefined;
    }
    return payload;
  }

  /* ── Post-login routing ─────────────────────────────────── */
  function _onLoginSuccess(data) {
    hideAuth();
    const role = data.user?.role || Auth.getRole();
    if (role === 'doctor') {
      DoctorDashboard.init();
    } else {
      PatientDashboard.init();
    }
  }

  /* ── Demo credentials ────────────────────────────────────  */
  function fillDemo(role) {
    if (role === 'doctor') {
      document.getElementById('loginEmail').value    = 'doctor@neurotrace.ai';
      document.getElementById('loginPassword').value = 'Doctor123!';
    } else {
      document.getElementById('loginEmail').value    = 'james.h@neurotrace.ai';
      document.getElementById('loginPassword').value = 'Patient123!';
    }
  }

  /* ── Alert helper ────────────────────────────────────────── */
  function _alert(id, msg, isError) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!msg) { el.classList.remove('visible'); return; }
    el.textContent = msg;
    el.className   = `auth-alert visible ${isError ? 'error' : 'success'}`;
  }

  return { showLogin, showSignup, hideAuth, fillDemo, _backToStep1 };
})();