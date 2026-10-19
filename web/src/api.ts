export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

export async function apiFetch(input: string, init: RequestInit = {}) {
  const url = new URL(input, API_BASE_URL);
  const headers = new Headers(init.headers);
  if (url.origin === new URL(API_BASE_URL).origin) {
    const token = localStorage.getItem("fatura-auth-token");
    if (token) headers.set("Authorization", `Bearer ${token}`);
    url.searchParams.delete("token");
  }
  try {
    return await window.fetch(url, { ...init, headers });
  } catch {
    throw new Error("Serveur inaccessible. Il peut être en cours de réveil ; réessayez dans une minute.");
  }
}
