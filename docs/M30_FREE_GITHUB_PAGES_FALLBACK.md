# Free GitHub Pages fallback for M30 Push-up local preview

## Why this exists

The owner asked for a free public Push-up web preview. GitHub PR
#55 added the browser-only 2D Push-up elbow angle and rolling
10-second local range. Python source good/bad model scoring is
**not** exposed for Push-up because the model has not achieved
reliable independent good/bad validation.

Vercel refused a direct production deployment on 27 October 2026
with HTTP **402 / api-deployments-young-hobby-team-24h**. A
preview build for feature commit `a6a54471` was READY, but
the Vercel project has SSO protection enabled for previews.
Do not disable Vercel security protection or add a paid plan
merely to show the research preview.

This free GitHub Pages workflow hosts ONLY static frontend
artifacts from the repository's existing `web` Vite app. It
does not deploy Python or any private Hugging Face models.

## One-time owner setup

GitHub requires that the repository administrator first enables
the GitHub Pages publishing source:

1. Open `https://github.com/idozii/Gym-Pose-Detection-System/settings/pages`.
2. Under **Build and deployment**, change **Source** to
   **GitHub Actions**. Save the setting.
3. Open **Actions** and select the workflow
   **Publish Free Push-up Pose Preview (GitHub Pages)**.
4. Run the workflow manually from **main** (or trigger a new
   qualifying frontend commit). After the deployment reports
   success, obtain the **real** page URL from the job output
   or Settings - Pages.

For a public repository this uses the GitHub Free
Pages/Actions offering. Availability and quota policies are
controlled by GitHub; the workflow adds no paid service.

The likely path is
`https://idozii.github.io/Gym-Pose-Detection-System/`
but **do not advertise it as live until GitHub Pages confirms
deployment succeeded**.

## What users can do

- Open a phone HTTPS URL and allow the physical camera in their
  own browser.
- Select **Push-up - Local Preview**.
- Observe pose overlay, elbow angle and 10s observed angle range
  without uploading a video, storing camera frames, counting reps
  or receiving a correct/wrong quality label.
- Switch front/rear camera and mirror only the selfie preview.

The GitHub Pages build explicitly sets
`VITE_INFERENCE_API_URL` to empty. That means all remote
research source-error scoring is **disabled** on this
standalone free static mirror, even for the existing
Squat/OHP/Row selectors. This avoids publishing any
production API URL or bypassing consent controls on an
unreviewed mirror. The existing Vercel site is unchanged.

The workflow compiles with Vite
`--base /Gym-Pose-Detection-System/` so all static assets
work at GitHub Pages' project subpath. A Vite base-aware
brand/home link also works at both the GitHub Pages subpath
and the existing Vercel root domain.

## Troubleshooting

If `configure-pages` errors that Pages is not enabled,
complete the one-time setting above. The workflow does not
possess a privileged token and cannot automatically enable it.

If the workflow build fails, check the GitHub Actions logs,
especially `Frontend unit tests` and `Compile with
project-relative public base path` steps. Use only the
published `web/dist` output, not the full Git repository,
as the Pages artifact.

The source media and all artifact/model/joblib/manifest
research files remain in private Git-ignored paths on the
owner's Ubuntu computer. A public static preview is not
evidence of model reliability or training success.
