import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createMemoryHistory, createRootRoute, createRoute, createRouter, Outlet, RouterProvider } from '@tanstack/react-router'
import { SidebarProvider } from '@/components/ui/sidebar'
import { Route as BrainRoute } from '@/routes/brain'
import { invoke } from '@tauri-apps/api/core'
import { ModelNav } from './Sidebar'

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(), Channel: class {} }))
vi.mock('@/containers/HeaderPage', () => ({ default: ({ children }: { children: React.ReactNode }) => children }))
vi.mock('./parts', () => ({ useTr: () => (key: string) => key }))
vi.mock('@/hooks/useAppUpdater', () => ({ useAppUpdater: vi.fn() }))
beforeEach(() => {
  vi.mocked(invoke).mockResolvedValue({ installed: { verified: false, capabilities: [] }, job: null })
})
afterEach(() => { cleanup(); vi.clearAllMocks() })

it.each(['/','/brain'])('opens the generated Brain route from %s without switching engines', async start => {
  const root = createRootRoute({ component: () => <SidebarProvider><ModelNav /><Outlet /></SidebarProvider> })
  const home = createRoute({ getParentRoute: () => root, path: '/', component: () => <p>Home</p> })
  const brain = BrainRoute.update({ path: '/brain', id: '/brain', getParentRoute: () => root } as never)
  const history = createMemoryHistory({ initialEntries: [start] })
  const router = createRouter({ history, routeTree: root.addChildren([home, brain]) })
  render(<RouterProvider router={router} />)
  const link = await screen.findByRole('link', { name: 'brain.nav' })
  expect(link).toHaveAttribute('href', '/brain')
  if (start === '/') fireEvent.click(link)
  await screen.findByRole('button', { name: 'brain.install' })
  await waitFor(() => expect(router.state.location.pathname).toBe('/brain'))
  expect(link).toHaveAttribute('data-active', 'true')
  expect(invoke).not.toHaveBeenCalledWith('brain_host_mode', expect.anything())
})
