// Schlichte Strich-Icons (24er Raster), ohne zusätzliche Bibliothek.
import type { ReactNode } from "react";

function I({ children, size = 18, title }: { children: ReactNode; size?: number; title?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden={title ? undefined : true}>
      {title && <title>{title}</title>}
      {children}
    </svg>
  );
}

type P = { size?: number; title?: string };
export const IcSliders = (p: P) => <I {...p}><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></I>;
export const IcCrop = (p: P) => <I {...p}><path d="M6 2v14a2 2 0 0 0 2 2h14" /><path d="M18 22V8a2 2 0 0 0-2-2H2" /></I>;
export const IcMask = (p: P) => <I {...p}><circle cx="9" cy="12" r="6" /><path d="M15 6.5a6 6 0 0 1 0 11" strokeDasharray="2 2" /><circle cx="15" cy="12" r="6" opacity=".45" /></I>;
export const IcPipette = (p: P) => <I {...p}><path d="m14.5 6.5 3 3" /><path d="M17.8 3.2a2.1 2.1 0 0 1 3 3l-2.3 2.3-3-3z" /><path d="M15.5 6.5 5 17l-1 4 4-1L18.5 9.5" /></I>;
export const IcUndo = (p: P) => <I {...p}><path d="M9 14 4 9l5-5" /><path d="M4 9h11a5 5 0 0 1 0 10h-3" /></I>;
export const IcRedo = (p: P) => <I {...p}><path d="m15 14 5-5-5-5" /><path d="M20 9H9a5 5 0 0 0 0 10h3" /></I>;
export const IcCompare = (p: P) => <I {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M12 4v16" /><path d="M3 15l4-4 5 5" opacity=".6" /></I>;
export const IcZoom = (p: P) => <I {...p}><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4M11 8v6M8 11h6" /></I>;
export const IcEye = (p: P) => <I {...p}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></I>;
export const IcTrash = (p: P) => <I {...p}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></I>;
export const IcPlus = (p: P) => <I {...p}><path d="M12 5v14M5 12h14" /></I>;
export const IcBrush = (p: P) => <I {...p}><path d="M18.4 2.6a2 2 0 0 1 2.9 2.9L12 14.8 9.2 12z" /><path d="M9 12.2c-2.2 0-4 1.8-4 4 0 1.7-1 2.6-2 3 1 .9 2.6 1.4 4 1.4 3 0 6-2.3 6-5.3z" /></I>;
export const IcLinear = (p: P) => <I {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M3 15h18" strokeDasharray="2 2" /></I>;
export const IcRadial = (p: P) => <I {...p}><ellipse cx="12" cy="12" rx="9" ry="7" /><ellipse cx="12" cy="12" rx="4.5" ry="3.5" strokeDasharray="2 2" /></I>;
export const IcSubject = (p: P) => <I {...p}><circle cx="12" cy="7" r="3.2" /><path d="M5.5 21c.6-4.2 3.2-6.5 6.5-6.5s5.9 2.3 6.5 6.5" /><path d="M2 2h4M2 2v4M22 2h-4M22 2v4M2 22h4M2 22v-4M22 22h-4M22 22v-4" /></I>;
export const IcSky = (p: P) => <I {...p}><path d="M7 18a4 4 0 0 1-.5-8 5.5 5.5 0 0 1 10.6-1.2A4.5 4.5 0 0 1 17.5 18z" /></I>;
export const IcBackground = (p: P) => <I {...p}><rect x="2.5" y="3.5" width="19" height="17" rx="2" /><circle cx="12" cy="10" r="2.5" fill="currentColor" opacity=".35" /><path d="M7.5 20.5c.5-3 2.3-4.8 4.5-4.8s4 1.8 4.5 4.8" fill="currentColor" opacity=".35" /></I>;
export const IcPeople = (p: P) => <I {...p}><circle cx="9" cy="8" r="3" /><circle cx="17" cy="9" r="2.4" /><path d="M3 20c.4-3.6 2.8-5.6 6-5.6s5.6 2 6 5.6M15 14.6c3 .1 5 1.9 5.4 5" /></I>;
export const IcCopy = (p: P) => <I {...p}><rect x="8" y="8" width="13" height="13" rx="2" /><path d="M4 16V5a2 2 0 0 1 2-2h11" /></I>;
export const IcPaste = (p: P) => <I {...p}><rect x="5" y="4" width="14" height="18" rx="2" /><path d="M9 4V2.5h6V4M9 11h6M9 15h4" /></I>;
export const IcReset = (p: P) => <I {...p}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /></I>;
export const IcSync = (p: P) => <I {...p}><path d="M20 11a8 8 0 0 0-14.3-4.9L4 8" /><path d="M4 3v5h5" /><path d="M4 13a8 8 0 0 0 14.3 4.9L20 16" /><path d="M20 21v-5h-5" /></I>;
export const IcBack = (p: P) => <I {...p}><path d="m15 18-6-6 6-6" /></I>;
export const IcGrid = (p: P) => <I {...p}><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></I>;
export const IcWand = (p: P) => <I {...p}><path d="m3 21 12-12M14 4l1-2 1 2 2 1-2 1-1 2-1-2-2-1zM19 10l.6-1.2L21 8.2l-1.4-.6L19 6.2l-.6 1.4-1.4.6 1.4.6zM8 3l.5-1L9 3l1 .5-1 .5L8.5 5 8 4l-1-.5z" /></I>;
export const IcCheck = (p: P) => <I {...p}><path d="m5 12 5 5L20 7" /></I>;
export const IcHistory = (p: P) => <I {...p}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5M12 7v5l3 2" /></I>;
export const IcRotate = (p: P) => <I {...p}><path d="M21 12a9 9 0 1 1-3-6.7L21 8" /><path d="M21 3v5h-5" /></I>;
export const IcClip = (p: P) => <I {...p}><path d="M4 20 12 4l8 16z" /></I>;
export const IcPreset = (p: P) => <I {...p}><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M7 15l3-3 2 2 5-5" /></I>;
export const IcInvert = (p: P) => <I {...p}><circle cx="12" cy="12" r="9" /><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" /></I>;
export const IcErase = (p: P) => <I {...p}><path d="m7 21-4-4 10-10 7 7-7 7zM21 21H7M9 11l7 7" /></I>;
export const IcHeal = (p: P) => <I {...p}><rect x="2.5" y="8.5" width="19" height="7" rx="3.5" transform="rotate(-45 12 12)" /><path d="M10.5 10.5h.01M13.5 13.5h.01M13.5 10.5h.01M10.5 13.5h.01" strokeWidth="2.4" /></I>;
