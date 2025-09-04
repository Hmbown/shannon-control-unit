# SCU Outreach Roadmap

## 📋 Overview
Strategic outreach campaign to secure compute partnership for 7B-70B validation of Shannon Control Unit.

## 🎯 Primary Goal
Secure 16-32 H100s for 72-96 hour pilot → Scale to 128 H100s for 30-day 70B validation

---

## Phase 1: Document Preparation (Immediate)

### 1.1 Two-Page Pilot Protocol ✅ HIGH PRIORITY
**File:** `documents/pilot_protocol_2pg.pdf`

**Page 1: Executive Summary**
- **Hero Result:** 15.6% perplexity reduction on Llama-3.2-1B
- **Value Prop:** 10-15% faster time-to-target = $100M+ savings at scale
- **Method:** One-line explanation (PI controller for automatic λ adjustment)
- **Ask:** Specific compute requirements (16-32 H100s, 72-96 hours)

**Page 2: Technical Validation**
- **Metrics Table:** 
  - Primary: Time-to-target perplexity vs tuned baseline
  - Secondary: Step-time overhead (<1-2%), variance across seeds
- **Risk Mitigation:** Why this is low-risk (additive, can disable anytime)
- **Timeline:** 3 phases with clear go/no-go gates
- **Deliverables:** Public case study, co-authorship, repro package

### 1.2 One-Pager Summary
**File:** `documents/one_pager.pdf`

**Sections:**
- Problem: Manual hyperparameter tuning wastes compute
- Solution: SCU automated control loop
- Proof: 15.6% improvement + plots
- Ask: Pilot partnership terms
- Contact: Direct booking link + email

### 1.3 Technical Appendix (Optional)
**File:** `documents/technical_appendix.pdf`
- Full MDL derivation
- Control theory background
- Ablation studies from 1B
- Patent claims summary

---

## Phase 2: Visual Assets

### 2.1 Hero Plot PNG ✅ CRITICAL
**File:** `visuals/scu_results_hero.png`

**4-panel figure (2x2):**
1. S(t) tracking at 1% ± 0.2pp
2. λ(t) adaptation over steps  
3. Validation BPT: Base vs SCU
4. 3B early results teaser

**Design notes:**
- Clean white background
- Sans-serif fonts (IBM Plex)
- Brand blue (#0052E0) for SCU line
- Include subtle grid, clear labels

### 2.2 Slide Deck (5 slides max)
**File:** `visuals/scu_pilot_deck.pdf`
1. Title + 15.6% hero stat
2. How it works (visual diagram)
3. Results table + plots
4. Pilot phases
5. Contact/next steps

---

## Phase 3: Email Templates

### 3.1 Cold Outreach - BD Leads
**File:** `templates/email_bd_cold.md`

```
Subject: 15% faster LLM training - seeking 7B pilot partner

Hi [Name],

We achieved 15.6% perplexity reduction on Llama-3.2-1B using the Shannon Control Unit—a closed-loop controller that eliminates manual hyperparameter search.

At [Company]'s scale, 10-15% efficiency = $XXM annual savings.

Can we discuss a low-risk 7B pilot? We need 16-32 H100s for 72-96 hours.

[Book 30min] | [View 2-pager]

Best,
Hunter
```

### 3.2 Warm Intro - Research Leads  
**File:** `templates/email_research_warm.md`

### 3.3 Follow-up Sequence
**File:** `templates/email_followup.md`
- Day 3: Add early 3B results
- Day 7: Share case study draft
- Day 14: Final ask with deadline

---

## Phase 4: Outreach Tracking

### 4.1 Target List
**File:** `tracking/targets.csv`

| Company | Contact | Role | Priority | Status | Notes |
|---------|---------|------|----------|--------|-------|
| OpenAI | [Name] | Research Lead | High | Not contacted | Via [mutual] |
| Anthropic | [Name] | BD | High | Email sent | Replied - scheduling |
| Meta FAIR | [Name] | Eng Manager | Medium | Warm intro | [Person] connecting |
| Google DeepMind | [Name] | PM | Medium | Not contacted | HN thread |

### 4.2 Response Tracking
**File:** `tracking/responses.md`
- Track: Opens, replies, meetings scheduled
- Key objections and how to address
- Commitment levels (verbal → written → signed)

---

## Phase 5: HN/Public Launch Strategy

### 5.1 HN Post Trigger Conditions
**DO NOT POST UNTIL:**
- [ ] Compute partnership secured (verbal minimum)
- [ ] 7B run shows clean Δ with profiler traces
- [ ] Patent provisional filed ✅ 
- [ ] Website live with plots ✅

### 5.2 Launch Sequence
1. **Pre-launch:** Seed with 3-5 friendly upvotes
2. **Launch:** Post at 9am PT Tuesday/Wednesday
3. **Engage:** Respond to every technical question
4. **Convert:** Funnel to Calendly → pilot discussions

### 5.3 Backup Plans
- If no compute by [date]: Consider cloud credits
- If poor HN reception: Focus on direct outreach
- If high interest: Prepare FAQ, screening criteria

---

## 📊 Success Metrics

### Week 1 Goals
- [ ] Send 10 targeted emails
- [ ] Book 3 discovery calls
- [ ] Get 1 verbal interest

### Week 2 Goals  
- [ ] Complete 5 technical deep-dives
- [ ] Receive 2 written proposals
- [ ] Start 1 pilot

### Month 1 Target
- [ ] 7B pilot complete with results
- [ ] 70B partnership secured
- [ ] Public case study drafted

---

## 🔧 Tools & Resources

### Required
- **Email:** Hunter.ai for finding contacts
- **CRM:** Simple spreadsheet or Notion
- **Calendar:** Calendly already set up ✅
- **Docs:** Google Docs for collaboration

### Nice to Have
- **Design:** Canva Pro for one-pager
- **Analytics:** Email tracking (Mailtrack)
- **Automation:** Sequences (Apollo.io free tier)

---

## 🚨 Risk Mitigation

### Technical Risks
- **7B doesn't replicate:** Have 3B results as backup
- **Overhead too high:** Pre-profile everything
- **Integration issues:** Offer hosted solution

### Business Risks
- **No compute interest:** Pivot to cloud providers
- **IP concerns:** Clear patent provisional filed
- **Competition:** Move fast, sign NDAs

---

## 📅 Timeline

### This Week
- [ ] Finalize 2-pager and one-pager
- [ ] Create hero PNG from existing plots
- [ ] Draft 3 email variants
- [ ] Identify 10 target contacts

### Next Week
- [ ] Send first batch of emails
- [ ] Iterate based on responses
- [ ] Prep for technical calls
- [ ] Update with 3B results

### Two Weeks Out
- [ ] Follow up on all leads
- [ ] Close verbal commitment
- [ ] Start pilot prep
- [ ] Consider HN if ready

---

## Notes
- Keep all materials version controlled
- Every email should have clear CTA
- Track everything - what works/doesn't
- Be ready to pivot messaging based on feedback