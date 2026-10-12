/* Planny web client: hash-routed single page over the planner API. No build step. */
"use strict";

(() => {
  // ---------------------------------------------------------------- constants

  const SLOTS = ["morning", "afternoon", "evening"];
  const POLL_MS = 2500;
  const MAX_INTERESTS = 15;
  const MAX_TRIP_DAYS = 30;
  const RECENT_KEY = "planny.recentPlans";
  const COUNTRY_KEY = "planny.country";

  const FIRST_RUN_STEPS = [
    ["validating request", "Checking the destination"],
    ["researching destination", "Researching sights, weather, entry rules and prices"],
    ["drafting itinerary", "Drafting the day-by-day plan"],
    ["preparing review", "Preparing your draft for review"],
  ];
  const REVISION_STEPS = [
    ["interpreting feedback", "Reading your feedback"],
    ["researching destination", "Researching again where needed"],
    ["drafting itinerary", "Revising the plan"],
    ["preparing review", "Preparing the revised draft"],
  ];
  const NODE_LABELS = {
    validate: "Destination check",
    research_agent: "Research",
    planner_agent: "Drafting",
    human_review: "Waiting for review",
    apply_patch: "Applying edits",
    route_feedback: "Reading feedback",
    finalize: "Finalizing",
  };

  // ---------------------------------------------------------------- state

  const state = {
    pollTimer: null,
    routeToken: 0,
    countries: [],
    pending: new Map(), // "day:slot" -> patch, for the plan under review
    pendingPlan: null,
  };

  const $view = document.getElementById("view");
  const $toast = document.getElementById("toast");

  // ---------------------------------------------------------------- utilities

  const esc = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const localeFor = (currency) => (currency === "INR" ? "en-IN" : undefined);

  function money(amount, currency) {
    if (amount == null || Number.isNaN(Number(amount))) return "—";
    try {
      return new Intl.NumberFormat(localeFor(currency), {
        style: currency ? "currency" : "decimal",
        currency: currency || undefined,
        maximumFractionDigits: 0,
      }).format(amount);
    } catch {
      return `${Math.round(amount).toLocaleString()} ${currency || ""}`.trim();
    }
  }

  function currencySymbol(currency) {
    try {
      const part = new Intl.NumberFormat(localeFor(currency), { style: "currency", currency })
        .formatToParts(0)
        .find((p) => p.type === "currency");
      return part ? part.value : currency;
    } catch {
      return "";
    }
  }

  function parseDate(iso) {
    const [y, m, d] = String(iso).split("-").map(Number);
    return new Date(y, m - 1, d);
  }

  const fmtDate = (iso, opts = { day: "numeric", month: "short", year: "numeric" }) =>
    iso ? new Intl.DateTimeFormat("en-GB", opts).format(parseDate(iso)) : "";

  const fmtDateTime = (iso) =>
    iso ? new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short" }).format(new Date(iso)) : "";

  function isoDate(date) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  }

  function addDays(date, n) {
    const d = new Date(date);
    d.setDate(d.getDate() + n);
    return d;
  }

  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  function storage(key, fallback) {
    try {
      const raw = window.localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch {
      return fallback;
    }
  }

  function store(key, value) {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable: a convenience only */
    }
  }

  function rememberPlan(id, destination) {
    const recent = storage(RECENT_KEY, []).filter((p) => p.id !== id);
    recent.unshift({ id, destination, at: new Date().toISOString() });
    store(RECENT_KEY, recent.slice(0, 8));
  }

  let toastTimer = null;
  function toast(message) {
    $toast.textContent = message;
    $toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      $toast.hidden = true;
    }, 5000);
  }

  // ---------------------------------------------------------------- API

  class ApiError extends Error {
    constructor(status, detail) {
      super(typeof detail === "string" ? detail : detail?.message || `Request failed (${status})`);
      this.status = status;
      this.detail = detail;
    }
  }

  async function api(path, { method = "GET", body } = {}) {
    let response;
    try {
      response = await fetch(path, {
        method,
        headers: body ? { "content-type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch {
      throw new ApiError(0, "Cannot reach the server. Check that it is running.");
    }
    const text = await response.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = text;
    }
    if (!response.ok) throw new ApiError(response.status, data?.detail ?? data);
    return data;
  }

  // ---------------------------------------------------------------- routing

  function stopPolling() {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function route() {
    stopPolling();
    const token = ++state.routeToken;
    const hash = window.location.hash.replace(/^#/, "") || "/";
    const plan = hash.match(/^\/plan\/([a-f0-9]{32})(\/history)?$/);
    if (hash === "/" || hash === "") return renderNewTrip();
    if (plan && plan[2]) return renderHistory(plan[1], token);
    if (plan) return renderPlan(plan[1], token);
    return renderNotFound("That page doesn't exist.");
  }

  const isCurrent = (token) => token === state.routeToken;

  function show(html) {
    $view.onclick = null;
    $view.onsubmit = null;
    $view.innerHTML = html;
    $view.focus({ preventScroll: true });
  }

  // ---------------------------------------------------------------- new trip

  function renderNewTrip() {
    document.title = "Planny · New trip";
    const today = new Date();
    const country = storage(COUNTRY_KEY, "India");
    show(`
      <p class="eyebrow">Begin a new journey</p>
      <h1 class="display">Where should we take you?</h1>
      <p class="lede">Tell us the essentials. Planny will research the details and bring you a complete draft to review.</p>
      <hr class="rule">
      <form class="form" id="trip-form" novalidate>
        <div class="field" data-field="destination">
          <label class="label" for="destination">Destination</label>
          <input class="input" id="destination" name="destination" required maxlength="120"
                 placeholder="e.g. Lisbon, Portugal" autocomplete="off">
        </div>
        <div class="field" data-field="origin">
          <label class="label" for="origin">Travelling from <span class="optional">(optional)</span></label>
          <input class="input" id="origin" name="origin" maxlength="120" placeholder="e.g. Delhi" autocomplete="off">
          <p class="hint">Add it and we'll plan and budget the trip there and back.</p>
        </div>
        <div class="field" data-field="dates">
          <span class="label" id="dates-label">Travel dates</span>
          <div class="dates" role="group" aria-labelledby="dates-label">
            <label><span class="mini">Start</span><input type="date" name="start_date" value="${isoDate(addDays(today, 14))}" required></label>
            <span class="arrow" aria-hidden="true">→</span>
            <label><span class="mini">End</span><input type="date" name="end_date" value="${isoDate(addDays(today, 17))}" required></label>
          </div>
        </div>
        <div class="form-row">
          <div class="field" data-field="budget_min">
            <label class="label" for="budget_min">Minimum budget</label>
            <div class="with-prefix"><span class="prefix" data-symbol></span>
              <input class="input money" id="budget_min" name="budget_min" type="number" min="1" step="1" inputmode="numeric" value="40000" required></div>
          </div>
          <div class="field" data-field="budget_max">
            <label class="label" for="budget_max">Maximum budget</label>
            <div class="with-prefix"><span class="prefix" data-symbol></span>
              <input class="input money" id="budget_max" name="budget_max" type="number" min="1" step="1" inputmode="numeric" value="80000" required></div>
          </div>
        </div>
        <div class="form-row">
          <div class="field" data-field="currency">
            <label class="label" for="currency">Currency</label>
            <input class="input" id="currency" name="currency" maxlength="3" autocomplete="off" data-upper>
            <p class="hint">Defaults to your country's currency</p>
          </div>
          <div class="field" data-field="travelers">
            <span class="label" id="travelers-label">Travellers</span>
            <div class="stepper" role="group" aria-labelledby="travelers-label">
              <button type="button" data-step="-1" aria-label="Fewer travellers">−</button>
              <output id="travelers" aria-live="polite">2</output>
              <button type="button" data-step="1" aria-label="More travellers">+</button>
            </div>
          </div>
        </div>
        <div class="field" data-field="nationality">
          <label class="label" for="nationality">Your country / passport</label>
          <input class="input" id="nationality" name="nationality" list="country-list" value="${esc(country)}" autocomplete="off">
          <datalist id="country-list"></datalist>
          <p class="hint">Used for visa and entry rules and your currency</p>
        </div>
        <div class="field" data-field="interests">
          <label class="label" for="interest-input">Interests</label>
          <div class="chips" id="chips"><input id="interest-input" placeholder="Type and press Enter" maxlength="40" autocomplete="off"></div>
          <p class="hint">Add up to ${MAX_INTERESTS}</p>
        </div>
        <div class="field" data-field="notes">
          <label class="label" for="notes">Notes</label>
          <textarea class="textarea" id="notes" name="notes" maxlength="1000"
                    placeholder="Anything we should know: vegetarian, travelling with parents, slow mornings…"></textarea>
        </div>
        <div class="form-actions">
          <button class="btn" type="submit" id="submit-trip">Plan my trip →</button>
          <button class="link-button" type="button" data-action="open-plan">Open an existing plan</button>
        </div>
      </form>`);
    wireTripForm();
  }

  function wireTripForm() {
    const form = document.getElementById("trip-form");
    const interests = ["Food", "History"];
    let travelers = 2;

    const chips = document.getElementById("chips");
    const chipInput = document.getElementById("interest-input");
    const drawChips = () => {
      chips.querySelectorAll(".chip").forEach((c) => c.remove());
      interests.forEach((label, i) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.innerHTML = `${esc(label)}<button type="button" aria-label="Remove ${esc(label)}" data-i="${i}">×</button>`;
        chips.insertBefore(chip, chipInput);
      });
      chipInput.disabled = interests.length >= MAX_INTERESTS;
    };
    const addInterest = () => {
      const value = chipInput.value.replace(/,/g, " ").trim();
      if (value && interests.length < MAX_INTERESTS && !interests.some((x) => x.toLowerCase() === value.toLowerCase())) {
        interests.push(value);
      }
      chipInput.value = "";
      drawChips();
    };
    chipInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") {
        e.preventDefault();
        addInterest();
      } else if (e.key === "Backspace" && !chipInput.value && interests.length) {
        interests.pop();
        drawChips();
      }
    });
    chipInput.addEventListener("blur", addInterest);
    chips.addEventListener("click", (e) => {
      const i = e.target.dataset?.i;
      if (i !== undefined) {
        interests.splice(Number(i), 1);
        drawChips();
        chipInput.focus();
      } else {
        chipInput.focus();
      }
    });
    drawChips();

    const out = document.getElementById("travelers");
    form.querySelectorAll("[data-step]").forEach((btn) =>
      btn.addEventListener("click", () => {
        travelers = Math.min(20, Math.max(1, travelers + Number(btn.dataset.step)));
        out.textContent = travelers;
        form.querySelector('[data-step="-1"]').disabled = travelers <= 1;
        form.querySelector('[data-step="1"]').disabled = travelers >= 20;
      }),
    );
    form.querySelector('[data-step="-1"]').disabled = travelers <= 1;

    const currencyInput = form.currency;
    const countryInput = form.nationality;
    const syncCurrency = () => {
      const typed = currencyInput.value.trim().toUpperCase();
      const match = findCountry(countryInput.value);
      const effective = typed || match?.currency || "USD";
      currencyInput.placeholder = `${match?.currency || "USD"} (auto)`;
      form.querySelectorAll("[data-symbol]").forEach((el) => {
        el.textContent = currencySymbol(effective);
      });
    };
    currencyInput.addEventListener("input", syncCurrency);
    countryInput.addEventListener("input", syncCurrency);
    loadCountries().then(() => {
      document.getElementById("country-list").innerHTML = state.countries
        .map((c) => `<option value="${esc(c.name)}">`)
        .join("");
      syncCurrency();
    });
    syncCurrency();

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      addInterest();
      const body = {
        destination: form.destination.value.trim(),
        origin: form.origin.value.trim() || null,
        start_date: form.start_date.value,
        end_date: form.end_date.value,
        budget_min: Number(form.budget_min.value),
        budget_max: Number(form.budget_max.value),
        currency: currencyInput.value.trim().toUpperCase() || null,
        travelers,
        nationality: countryInput.value.trim() || null,
        interests: [...interests],
        notes: form.notes.value.trim() || null,
      };
      const errors = validateTrip(body);
      showFieldErrors(form, errors);
      if (Object.keys(errors).length) return;

      const button = document.getElementById("submit-trip");
      button.setAttribute("aria-busy", "true");
      button.disabled = true;
      try {
        const created = await api("/plan", { method: "POST", body });
        if (body.nationality) store(COUNTRY_KEY, body.nationality);
        rememberPlan(created.plan_id, body.destination);
        window.location.hash = `#/plan/${created.plan_id}`;
      } catch (err) {
        if (err.status === 422 && Array.isArray(err.detail)) {
          showFieldErrors(form, serverFieldErrors(err.detail));
        } else {
          toast(err.message);
        }
      } finally {
        button.removeAttribute("aria-busy");
        button.disabled = false;
      }
    });
  }

  function validateTrip(t) {
    const errors = {};
    const today = isoDate(new Date());
    if (t.destination.length < 2) errors.destination = "Tell us where you'd like to go.";
    if (!t.start_date || !t.end_date) errors.dates = "Choose both travel dates.";
    else if (t.start_date < today) errors.dates = "The trip can't start in the past.";
    else if (t.end_date <= t.start_date) errors.dates = "The end date must be after the start date.";
    else if ((parseDate(t.end_date) - parseDate(t.start_date)) / 86400000 + 1 > MAX_TRIP_DAYS)
      errors.dates = `Trips can be at most ${MAX_TRIP_DAYS} days.`;
    if (!(t.budget_min > 0)) errors.budget_min = "Enter a minimum budget above zero.";
    if (!(t.budget_max > 0)) errors.budget_max = "Enter a maximum budget above zero.";
    else if (t.budget_max < t.budget_min) errors.budget_max = "The maximum must be at least the minimum.";
    if (t.currency && !/^[A-Z]{3}$/.test(t.currency)) errors.currency = "Use a 3-letter currency code, e.g. INR.";
    return errors;
  }

  function serverFieldErrors(detail) {
    const errors = {};
    for (const item of detail) {
      const field = item.loc?.[item.loc.length - 1];
      const key = ["start_date", "end_date"].includes(field) ? "dates" : field;
      const message = String(item.msg || "Invalid value").replace(/^Value error, /, "");
      if (key && key !== "body") errors[key] = message;
      else errors.dates = errors.dates || message; // cross-field rules (dates, budget) arrive on the body
    }
    return errors;
  }

  function showFieldErrors(form, errors) {
    form.querySelectorAll(".field").forEach((field) => {
      field.classList.remove("invalid");
      field.querySelector(".field-error")?.remove();
      const message = errors[field.dataset.field];
      if (message) {
        field.classList.add("invalid");
        field.insertAdjacentHTML("beforeend", `<p class="field-error" role="alert">${esc(message)}</p>`);
      }
    });
    const first = form.querySelector(".field.invalid input, .field.invalid textarea");
    first?.focus();
  }

  async function loadCountries() {
    if (state.countries.length) return;
    try {
      state.countries = await api("/countries");
    } catch {
      state.countries = [];
    }
  }

  function findCountry(text) {
    const v = String(text || "").trim().toLowerCase();
    return state.countries.find((c) => c.name.toLowerCase() === v || c.iso2.toLowerCase() === v);
  }

  // ---------------------------------------------------------------- plan

  async function renderPlan(id, token) {
    let plan;
    try {
      plan = await api(`/plan/${id}`);
    } catch (err) {
      if (!isCurrent(token)) return;
      if (err.status === 404) return renderNotFound("We couldn't find a plan with that ID.");
      return renderError(err.message, () => renderPlan(id, token));
    }
    if (!isCurrent(token)) return;
    rememberPlan(id, plan.request?.destination);
    if (state.pendingPlan !== id) {
      state.pending.clear();
      state.pendingPlan = id;
    }
    if (plan.status === "running") return renderRunning(plan, token);
    if (plan.status === "awaiting_review") return renderReview(plan);
    if (plan.status === "completed") return renderFinal(plan, token);
    return renderFailed(plan);
  }

  function schedule(id, token) {
    stopPolling();
    state.pollTimer = setTimeout(() => {
      if (isCurrent(token)) renderPlan(id, token);
    }, POLL_MS);
  }

  function tripLine(request) {
    if (!request) return "";
    const nights = Math.round((parseDate(request.end_date) - parseDate(request.start_date)) / 86400000);
    return [
      request.origin ? `${esc(request.origin)} → ${esc(request.destination)}` : esc(request.destination),
      `${fmtDate(request.start_date, { day: "numeric", month: "short" })} – ${fmtDate(request.end_date)}`,
      plural(nights, "night"),
      plural(request.travelers, "traveller"),
    ]
      .map((x) => `<span>${x}</span>`)
      .join("");
  }

  function runningHead(plan) {
    const revising = plan.revisions_used > 0;
    return `
      <p class="eyebrow">${revising ? `Revision ${plan.revisions_used}` : "Planning in progress"}</p>
      <h1 class="display">${revising ? "Reworking your plan" : `Planning ${esc(plan.request?.destination || "your trip")}`}</h1>
      <div class="plan-meta">${tripLine(plan.request)}</div>`;
  }

  function runningSteps(plan) {
    const steps = plan.revisions_used > 0 ? REVISION_STEPS : FIRST_RUN_STEPS;
    const at = steps.findIndex(([stage]) => stage === plan.stage);
    return steps
      .map(([, label], i) => {
        const cls = at < 0 ? (i === 0 ? "current" : "") : i < at ? "done" : i === at ? "current" : "";
        const note = i === at || (at < 0 && i === 0) ? "In progress" : i < at ? "Done" : "";
        return `<li class="${cls}"><span class="dot" aria-hidden="true"></span><span>${esc(label)}</span><span class="step-note">${note}</span></li>`;
      })
      .join("");
  }

  function steerLog(plan) {
    const notes = plan.steering || [];
    if (!notes.length) return "";
    return notes.map((n) => `<li>${esc(n.text)}</li>`).join("");
  }

  function renderRunning(plan, token) {
    document.title = `Planny · Planning ${plan.request?.destination || ""}`;
    const mounted = document.getElementById("running");
    // Later polls update in place, so a message being typed into the steer box survives.
    if (mounted?.dataset.plan === plan.plan_id) {
      document.getElementById("run-head").innerHTML = runningHead(plan);
      document.getElementById("run-steps").innerHTML = runningSteps(plan);
      document.getElementById("steer-log").innerHTML = steerLog(plan);
      schedule(plan.plan_id, token);
      return;
    }
    show(`
      <div id="running" data-plan="${esc(plan.plan_id)}">
        <div id="run-head">${runningHead(plan)}</div>
        <p class="lede">This usually takes two to five minutes. You can leave this page; the plan keeps going and is saved under its ID.</p>
        <ol class="steps" id="run-steps">${runningSteps(plan)}</ol>
        <section class="panel steer" aria-labelledby="steer-title">
          <form class="form" id="steer-form" novalidate>
            <div class="field" data-field="steer">
              <label class="label" id="steer-title" for="steer">Steer this plan</label>
              <textarea class="textarea" id="steer" maxlength="500" required rows="2"
                        placeholder="e.g. We're vegetarian · Keep evenings free · Make it Goa instead"></textarea>
              <p class="hint">No need to wait for the draft. Planny works out whether to adjust what it's doing now, go back a step, or start over.</p>
            </div>
            <div class="steer-actions">
              <p class="steer-status" id="steer-status" role="status"></p>
              <button class="btn" type="submit">Send guidance</button>
            </div>
          </form>
          <ul class="steer-log" id="steer-log" aria-label="Guidance sent">${steerLog(plan)}</ul>
        </section>
        <p class="hint" data-space="md">Plan ID <span class="num">${esc(plan.plan_id)}</span></p>
      </div>`);
    wireSteer(plan.plan_id, token);
    schedule(plan.plan_id, token);
  }

  const STEER_STAGE = { request: "trip details", research: "research", planner: "planning" };
  const STEER_OUTCOME = {
    live: () => "Passed to the agent working now. It adjusts from its next step.",
    queued: (r) => `Saved for the ${STEER_STAGE[r.scope]} step, which hasn't started yet.`,
    rewind: (r) => `Going back to ${STEER_STAGE[r.scope]} with this. Work before that is kept.`,
    restart: () => "That changes the trip itself, so planning starts over with the new details.",
  };

  function wireSteer(id, token) {
    $view.onsubmit = async (e) => {
      if (e.target.id !== "steer-form") return;
      e.preventDefault();
      const form = e.target;
      const input = form.querySelector("#steer");
      const status = form.querySelector("#steer-status");
      const button = form.querySelector("button[type=submit]");
      const message = input.value.trim();
      if (message.length < 2) {
        status.textContent = "Write a few words first.";
        input.focus();
        return;
      }
      button.disabled = true;
      status.textContent = "Working out where this fits…";
      try {
        const result = await api(`/plan/${id}/steer`, { method: "POST", body: { message } });
        input.value = "";
        status.textContent = STEER_OUTCOME[result.action](result);
      } catch (err) {
        status.textContent = err.message;
        if (err.status !== 409) return;
        toast(err.message);
      } finally {
        button.disabled = false;
      }
      if (isCurrent(token)) {
        stopPolling();
        renderPlan(id, token);
      }
    };
  }

  // ---------------------------------------------------------------- itinerary

  function effectiveSlot(day, slot) {
    const patch = state.pending.get(`${day.day}:${slot}`);
    if (!patch) return { activity: day[slot], pending: false };
    return { activity: patch.op === "replace" ? patch.value : null, pending: true };
  }

  function renderDay(day, currency, { editable }) {
    const slots = SLOTS.map((slot) => {
      const { activity: a, pending } = editable ? effectiveSlot(day, slot) : { activity: day[slot], pending: false };
      const controls = editable
        ? `<div class="no-print">
             <button class="btn quiet" type="button" data-edit="${day.day}:${slot}">${a ? "Edit" : "Add"}</button>
             ${a ? `<button class="btn quiet" type="button" data-remove="${day.day}:${slot}">Remove</button>` : ""}
             ${pending ? `<button class="btn quiet" type="button" data-undo="${day.day}:${slot}">Undo</button>` : ""}
           </div>`
        : "";
      const body = a
        ? `<p class="slot-title">${esc(a.name)}</p>${a.description ? `<p class="slot-desc">${esc(a.description)}</p>` : ""}`
        : `<p class="slot-title slot-empty">Free time</p>`;
      const cost = a && a.est_cost ? `<span class="money">${money(a.est_cost, currency)}</span>` : "";
      return `<div class="slot${pending ? " pending" : ""}">
          <span class="slot-name">${slot}</span>
          <div>${body}</div>
          <div class="slot-side">${cost}${controls}</div>
        </div>`;
    }).join("");

    const logistics = (day.logistics || []).length
      ? `<ul class="logistics" aria-label="Logistics">${day.logistics
          .map((l) => `<li>${esc(l.name)}${l.est_cost ? ` · <span class="money">${money(l.est_cost, currency)}</span>` : ""}</li>`)
          .join("")}</ul>`
      : "";

    return `<section class="day" aria-labelledby="day-${day.day}">
        <div class="day-head">
          <h3 id="day-${day.day}">Day ${day.day} · ${esc(day.theme)}</h3>
          <span class="day-date">${fmtDate(day.date, { weekday: "long", day: "numeric", month: "long" })}</span>
        </div>
        ${day.weather_note ? `<p class="day-weather">${esc(day.weather_note)}</p>` : ""}
        ${slots}
        ${logistics}
        <p class="day-costs">Lodging <span class="money">${money(day.lodging_cost, currency)}</span>
          · Food <span class="money">${money(day.food_cost, currency)}</span>
          · Local transport <span class="money">${money(day.transit_cost, currency)}</span></p>
      </section>`;
  }

  function renderItinerary(draft, request, { editable = false } = {}) {
    const b = draft.budget;
    const currency = b?.currency || request?.currency;
    const ledger = b
      ? `<dl class="ledger">
           <div><dt>Estimated total</dt><dd class="money${b.within_budget ? "" : " over"}">${money(b.total_estimated, currency)}</dd></div>
           <div><dt>Budget cap</dt><dd class="money">${money(b.cap, currency)}</dd></div>
           <div><dt>${b.remaining >= 0 ? "Left in budget" : "Over budget by"}</dt><dd class="money${b.remaining >= 0 ? "" : " over"}">${money(Math.abs(b.remaining), currency)}</dd></div>
         </dl>
         <ul class="categories">${Object.entries(b.by_category)
           .map(([k, v]) => `<li>${esc(k)} <b class="money">${money(v, currency)}</b></li>`)
           .join("")}</ul>`
      : "";

    const facts = [];
    if (draft.intercity) {
      const i = draft.intercity;
      facts.push(["Getting there", `${esc(i.summary)} · <span class="money">${money(i.cost_per_person_round_trip, currency)}</span> per person, round trip`]);
    }
    if (draft.entry_requirements) {
      facts.push(["Entry requirements", `${esc(draft.entry_requirements)}<p class="hint">Verify with the official source before booking.</p>`]);
    }
    const fx = draft.exchange_rate;
    if (fx && fx.base !== fx.quote) {
      facts.push(["Exchange rate", `1 ${esc(fx.base)} = <span class="num">${esc(fx.rate)}</span> ${esc(fx.quote)}${fx.date ? ` <span class="aside">(${fmtDate(fx.date)})</span>` : ""}`]);
    }

    const sources = (draft.sources || []).length
      ? `<h2 class="section-title sources-title">Sources</h2><ol class="sources">${draft.sources
          .map((s) => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a></li>`)
          .join("")}</ol>`
      : "";

    return `
      ${ledger}
      ${facts.length ? `<div class="facts">${facts.map(([k, v]) => `<div class="fact"><span class="label">${k}</span><div>${v}</div></div>`).join("")}</div>` : ""}
      <div class="days">
      ${draft.days.map((d) => renderDay(d, currency, { editable })).join("")}
      </div>
      ${(draft.caveats || []).length ? `<div class="facts"><div class="fact"><span class="label">Planner notes</span><div><ul class="tray-list">${draft.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div></div></div>` : ""}
      ${sources}`;
  }

  function notices(plan, draft) {
    const out = [];
    if ((draft.quality_issues || []).length) {
      out.push(`<div class="notice"><strong>Worth checking before you approve</strong><ul>${draft.quality_issues.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`);
    }
    const degraded = plan.degraded || [];
    if (degraded.length) {
      out.push(`<div class="notice info"><strong>Some information was unavailable</strong><ul>${degraded.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`);
    }
    return out.join("");
  }

  // ---------------------------------------------------------------- review

  function renderReview(plan) {
    const draft = plan.draft;
    const left = plan.review?.payload?.revisions_left;
    document.title = `Planny · Review ${draft.title}`;
    show(`
      <div class="plan-head">
        <p class="eyebrow">${plan.revisions_used ? `Revised draft · ${plural(plan.revisions_used, "revision")}` : "Draft for your review"}</p>
        <span class="status-pill status-awaiting_review">Awaiting review</span>
      </div>
      <h1 class="display">${esc(draft.title)}</h1>
      <div class="plan-meta">${tripLine(plan.request)}</div>
      <p class="lede">${esc(draft.summary)}</p>
      <div data-space="lg">${notices(plan, draft)}</div>
      ${renderItinerary(draft, plan.request, { editable: true })}
      <div class="review-dock" id="dock"></div>`);
    drawDock(plan, left);
    wireReview(plan);
  }

  function drawDock(plan, revisionsLeft) {
    const dock = document.getElementById("dock");
    const n = state.pending.size;
    if (n) {
      const items = [...state.pending.values()]
        .map((p) => `<li>Day ${p.day} ${p.slot}: ${p.op === "remove" ? "remove" : esc(p.value.name)}</li>`)
        .join("");
      dock.innerHTML = `
        <div class="dock-row">
          <div><strong>${plural(n, "pending change")}</strong><ul class="tray-list">${items}</ul></div>
          <div class="dock-actions">
            <button class="btn secondary" type="button" data-action="discard">Discard</button>
            <button class="btn" type="button" data-action="apply">Apply ${plural(n, "change")}</button>
          </div>
        </div>`;
      return;
    }
    dock.innerHTML = `
      <div class="dock-row">
        <span class="dock-note">Edit activities, ask for changes, or approve.${revisionsLeft != null ? ` ${plural(revisionsLeft, "revision")} left.` : ""}</span>
        <div class="dock-actions">
          <button class="btn secondary" type="button" data-action="feedback" aria-expanded="false" aria-controls="feedback-panel">Request changes</button>
          <button class="btn" type="button" data-action="approve">Approve plan</button>
        </div>
      </div>
      <div class="panel" id="feedback-panel" hidden>
        <form class="form" id="feedback-form" novalidate>
          <div class="field" data-field="feedback">
            <label class="label" for="feedback">What should change?</label>
            <textarea class="textarea" id="feedback" maxlength="2000" required
                      placeholder="e.g. Too much walking on day 2, and add a cooking class."></textarea>
          </div>
          <fieldset class="field bare">
            <legend class="label">Send it to</legend>
            <div class="radio-row">
              <label><input type="radio" name="route" value="" checked> Let Planny decide</label>
              <label><input type="radio" name="route" value="research"> Research again (wrong places or facts)</label>
              <label><input type="radio" name="route" value="planner"> Re-plan (pace, order, budget)</label>
            </div>
          </fieldset>
          <div class="dialog-actions">
            <button class="btn secondary" type="button" data-action="feedback-cancel">Cancel</button>
            <button class="btn" type="submit">Send feedback</button>
          </div>
        </form>
      </div>`;
  }

  function wireReview(plan) {
    const refresh = () => {
      const scroll = window.scrollY;
      renderReview(plan);
      window.scrollTo(0, scroll);
    };

    $view.onclick = async (e) => {
      const target = e.target.closest("button");
      if (!target) return;
      const { edit, remove, undo, action } = target.dataset;
      if (edit) return openActivityEditor(plan, edit, refresh);
      if (remove) {
        const [day, slot] = remove.split(":");
        state.pending.set(remove, { op: "remove", day: Number(day), slot });
        return refresh();
      }
      if (undo) {
        state.pending.delete(undo);
        return refresh();
      }
      if (action === "discard") {
        state.pending.clear();
        return refresh();
      }
      if (action === "apply") {
        return submitReview(plan, target, { action: "modify", patches: [...state.pending.values()] }, () => state.pending.clear());
      }
      if (action === "approve") return submitReview(plan, target, { action: "approve" });
      if (action === "feedback" || action === "feedback-cancel") {
        const panel = document.getElementById("feedback-panel");
        const opening = panel.hidden && action === "feedback";
        panel.hidden = !opening;
        document.querySelector('[data-action="feedback"]').setAttribute("aria-expanded", String(opening));
        if (opening) document.getElementById("feedback").focus();
      }
    };

    $view.onsubmit = (e) => {
      if (e.target.id !== "feedback-form") return;
      e.preventDefault();
      const form = e.target;
      const feedback = form.querySelector("#feedback").value.trim();
      if (!feedback) return showFieldErrors(form, { feedback: "Tell us what to change." });
      const route = form.querySelector('input[name="route"]:checked').value || null;
      submitReview(plan, form.querySelector('[type="submit"]'), { action: "reject", feedback, route_hint: route });
    };
  }

  async function submitReview(plan, button, body, onAccepted) {
    button.setAttribute("aria-busy", "true");
    button.disabled = true;
    try {
      await api(`/plan/${plan.plan_id}/review`, {
        method: "POST",
        body: { ...body, checkpoint_id: plan.checkpoint_id },
      });
      onAccepted?.();
      window.scrollTo(0, 0);
      const token = ++state.routeToken;
      renderPlan(plan.plan_id, token);
    } catch (err) {
      button.removeAttribute("aria-busy");
      button.disabled = false;
      if (err.status === 409) {
        toast("This plan changed since you opened it. Showing the latest version.");
        renderPlan(plan.plan_id, ++state.routeToken);
      } else if (err.status === 422) {
        toast(Array.isArray(err.detail) ? err.detail.map((d) => d.msg).join(" ") : err.message);
      } else {
        toast(err.message);
      }
    }
  }

  function openActivityEditor(plan, key, onSave) {
    const [dayNo, slot] = key.split(":");
    const day = plan.draft.days.find((d) => d.day === Number(dayNo));
    const current = effectiveSlot(day, slot).activity;
    const currency = plan.draft.budget?.currency || plan.request?.currency;
    const dialog = document.createElement("dialog");
    dialog.setAttribute("aria-labelledby", "edit-title");
    dialog.innerHTML = `
      <form method="dialog" class="dialog-body form" novalidate>
        <div>
          <p class="eyebrow">Day ${day.day} · ${esc(slot)}</p>
          <h2 class="section-title" id="edit-title">${current ? "Edit activity" : "Add an activity"}</h2>
        </div>
        <div class="field" data-field="name">
          <label class="label" for="act-name">Activity</label>
          <input class="input" id="act-name" required maxlength="200" value="${esc(current?.name || "")}" placeholder="e.g. Fado evening in Alfama">
        </div>
        <div class="field">
          <label class="label" for="act-desc">Description <span class="optional">(optional)</span></label>
          <textarea class="textarea" id="act-desc" maxlength="500">${esc(current?.description || "")}</textarea>
        </div>
        <div class="form-row">
          <div class="field" data-field="cost">
            <label class="label" for="act-cost">Cost for the group</label>
            <div class="with-prefix"><span class="prefix">${esc(currencySymbol(currency))}</span>
              <input class="input money" id="act-cost" type="number" min="0" step="1" value="${current?.est_cost ?? 0}"></div>
          </div>
          <div class="field" data-field="hours">
            <label class="label" for="act-hours">Hours</label>
            <input class="input num" id="act-hours" type="number" min="0.5" max="12" step="0.5" value="${current?.duration_hours ?? 2}">
          </div>
        </div>
        <div class="dialog-actions">
          <button class="btn secondary" value="cancel" formnovalidate>Cancel</button>
          <button class="btn" value="save">Add to changes</button>
        </div>
      </form>`;
    document.body.appendChild(dialog);
    dialog.addEventListener("close", () => dialog.remove());
    const form = dialog.querySelector("form");
    form.addEventListener("submit", (e) => {
      if (e.submitter?.value !== "save") return;
      const name = dialog.querySelector("#act-name").value.trim();
      const cost = Number(dialog.querySelector("#act-cost").value);
      const hours = Number(dialog.querySelector("#act-hours").value);
      const errors = {};
      if (!name) errors.name = "Name the activity.";
      if (!(cost >= 0)) errors.cost = "Enter zero or more.";
      if (!(hours > 0 && hours <= 12)) errors.hours = "Between 0.5 and 12 hours.";
      if (Object.keys(errors).length) {
        e.preventDefault();
        showFieldErrors(form, errors);
        return;
      }
      state.pending.set(key, {
        op: "replace",
        day: day.day,
        slot,
        value: {
          ...(current || {}),
          name,
          description: dialog.querySelector("#act-desc").value.trim(),
          est_cost: cost,
          duration_hours: hours,
        },
      });
      queueMicrotask(onSave);
    });
    dialog.showModal();
  }

  // ---------------------------------------------------------------- final

  async function renderFinal(plan, token) {
    let final;
    try {
      final = await api(`/plan/${plan.plan_id}/final`);
    } catch (err) {
      if (isCurrent(token)) renderError(err.message, () => renderPlan(plan.plan_id, ++state.routeToken));
      return;
    }
    if (!isCurrent(token)) return;
    const draft = final.itinerary;
    const followup = final.status === "needs_human_followup";
    document.title = `Planny · ${draft.title}`;
    show(`
      <div class="plan-head">
        <p class="eyebrow">${followup ? "Needs follow-up" : "Your itinerary"}</p>
        <span class="status-pill status-completed">${followup ? "Finalized" : "Approved"}</span>
      </div>
      <h1 class="display">${esc(draft.title)}</h1>
      <div class="plan-meta">${tripLine(plan.request)}</div>
      <p class="lede">${esc(draft.summary)}</p>
      <div data-space="lg">
        ${followup
          ? `<div class="notice"><strong>The revision limit was reached</strong>This is the last draft after ${plural(final.revisions, "revision")}. Someone should review it before booking.</div>`
          : `<div class="notice success"><strong>Approved ${esc(fmtDateTime(final.finalized_at))}</strong>${final.revisions ? `After ${plural(final.revisions, "revision")}.` : "Approved on the first draft."}</div>`}
        ${(final.degraded || []).length ? `<div class="notice info"><strong>Some information was unavailable while planning</strong><ul>${final.degraded.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` : ""}
      </div>
      ${renderItinerary(draft, plan.request)}
      <div class="form-actions no-print" data-space="xl">
        <div class="dock-actions">
          <button class="btn" type="button" data-action="print">Print itinerary</button>
          <a class="btn secondary" href="#/plan/${plan.plan_id}/history">View history</a>
        </div>
        <a href="#/">Plan another trip</a>
      </div>`);
    $view.onclick = (e) => {
      if (e.target.closest('[data-action="print"]')) window.print();
    };
  }

  // ---------------------------------------------------------------- failed / errors

  function renderFailed(plan) {
    document.title = "Planny · Planning stopped";
    show(`
      <div class="plan-head">
        <p class="eyebrow">Planning stopped</p>
        <span class="status-pill status-failed">Failed</span>
      </div>
      <h1 class="display">We couldn't finish this plan</h1>
      <div class="plan-meta">${tripLine(plan.request)}</div>
      <div class="notice error" data-space="lg"><strong>What went wrong</strong>${esc(plan.error || "The planner stopped unexpectedly.")}</div>
      <p class="lede">If you've fixed the cause (for example an API key), retry from the step that failed. Work already done is kept.</p>
      <div class="form-actions" data-space="md">
        <div class="dock-actions">
          <button class="btn" type="button" data-action="retry">Retry from the failed step</button>
          <a class="btn secondary" href="#/plan/${plan.plan_id}/history">View history</a>
        </div>
        <a href="#/">Start a new trip</a>
      </div>`);
    $view.onclick = async (e) => {
      const button = e.target.closest('[data-action="retry"]');
      if (!button) return;
      button.setAttribute("aria-busy", "true");
      button.disabled = true;
      try {
        await api(`/plan/${plan.plan_id}/retry`, { method: "POST" });
        renderPlan(plan.plan_id, ++state.routeToken);
      } catch (err) {
        button.removeAttribute("aria-busy");
        button.disabled = false;
        toast(err.message);
      }
    };
  }

  function renderNotFound(message) {
    document.title = "Planny · Not found";
    show(`
      <p class="eyebrow">Off the map</p>
      <h1 class="display">Nothing here</h1>
      <p class="lede">${esc(message)}</p>
      <div class="form-actions" data-space="lg">
        <a class="btn" href="#/">Plan a new trip</a>
        <button class="link-button" type="button" data-action="open-plan">Open an existing plan</button>
      </div>`);
  }

  function renderError(message, retry) {
    show(`
      <p class="eyebrow">Something went wrong</p>
      <h1 class="display">We hit a snag</h1>
      <div class="notice error" data-space="lg">${esc(message)}</div>
      <button class="btn" type="button" data-action="reload">Try again</button>`);
    $view.onclick = (e) => {
      if (e.target.closest('[data-action="reload"]')) retry();
    };
  }

  // ---------------------------------------------------------------- history

  async function renderHistory(id, token) {
    let history;
    let plan;
    try {
      [history, plan] = await Promise.all([api(`/plan/${id}/history`), api(`/plan/${id}`)]);
    } catch (err) {
      if (!isCurrent(token)) return;
      if (err.status === 404) return renderNotFound("We couldn't find a plan with that ID.");
      return renderError(err.message, () => renderHistory(id, token));
    }
    if (!isCurrent(token)) return;
    document.title = "Planny · Plan history";
    const rows = history.checkpoints
      .filter((c) => c.next.length || c.has_draft)
      .map((c) => {
        const what = c.next.length ? NODE_LABELS[c.next[0]] || c.next[0] : "Finished";
        return `<li><strong>${esc(what)}</strong>${c.revision_count ? ` · revision ${c.revision_count}` : ""}
          <div class="when">${esc(fmtDateTime(c.created_at))} · step ${esc(c.step)}</div></li>`;
      })
      .join("");
    show(`
      <p class="eyebrow">Plan history</p>
      <h1 class="display">${esc(plan.draft?.title || plan.request?.destination || "Your plan")}</h1>
      <div class="plan-meta">${tripLine(plan.request)}</div>
      <p class="lede">Every saved step of this plan, newest first.</p>
      ${rows ? `<ol class="timeline">${rows}</ol>` : `<p class="empty">No history yet.</p>`}
      <div class="form-actions" data-space="md"><a class="btn secondary" href="#/plan/${id}">Back to the plan</a></div>`);
  }

  // ---------------------------------------------------------------- open-plan dialog

  function openPlanDialog() {
    const dialog = document.getElementById("open-dialog");
    const recent = storage(RECENT_KEY, []);
    document.getElementById("recent-plans").innerHTML = recent.length
      ? `Recent: ${recent
          .slice(0, 5)
          .map((p) => `<a href="#/plan/${esc(p.id)}" data-recent>${esc(p.destination || p.id.slice(0, 8))}</a>`)
          .join(" · ")}`
      : "";
    document.getElementById("open-id").value = "";
    dialog.showModal();
  }

  function wireChrome() {
    document.getElementById("open-plan").addEventListener("click", openPlanDialog);
    document.addEventListener("click", (e) => {
      if (e.target.closest('[data-action="open-plan"]')) openPlanDialog();
      if (e.target.closest("[data-recent]")) document.getElementById("open-dialog").close();
    });
    const dialog = document.getElementById("open-dialog");
    document.getElementById("open-form").addEventListener("submit", (e) => {
      if (e.submitter?.value !== "open") return;
      const id = document.getElementById("open-id").value.trim().toLowerCase();
      if (!/^[a-f0-9]{32}$/.test(id)) {
        e.preventDefault();
        toast("A plan ID is 32 characters: letters a–f and digits.");
        return;
      }
      dialog.close();
      window.location.hash = `#/plan/${id}`;
    });
    window.addEventListener("hashchange", route);
  }

  wireChrome();
  route();
})();
