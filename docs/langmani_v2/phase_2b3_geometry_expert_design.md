# Phase 2B.3 geometry-constrained closed-loop Push expert

## Design boundary

`GeometryConstrainedPushExpert` is the single architecture-level replacement
authorized after the Candidate E--P failure audit. It does not wrap Candidate P,
reuse a recorded trajectory, mutate simulator state, modify success, expand the
workspace, or clip an invalid action. It retains only the validated native
MPlib/Panda planning and action-execution boundary from the earlier expert.

The expert consumes privileged simulator state only through explicit expert
accessors. These fields are unavailable to later SmolVLA observations. Formal
data collection, LeRobot export, optimizer work, Phase 2C.2, and Phase 2D remain
out of scope and at zero.

## Explicit state machine

The versioned semantic is `geometry_constrained_closed_loop_push_v1`. Every
transition is checked against an immutable adjacency map and recorded with the
current step, distance, recovery count, replan count, and observable reason.

| State | Entry | Control target | Progress | Bound | Failure routing |
| --- | --- | --- | --- | --- | --- |
| `PLAN_PUSH` | Fresh compatible reset | Complete geometry plan | Geometry and native planner initialized | 1 visit | `SAFE_ABORT` |
| `MOVE_TO_STAGING` | Accepted plan | Elevated safe staging pose | TCP reaches tolerance | 4 visits | `REPLAN` or `SAFE_ABORT` |
| `MOVE_TO_PRECONTACT` | Staging reached | Support-aware precontact pose | TCP reaches tolerance | 4 visits | `REPLAN` or `SAFE_ABORT` |
| `ESTABLISH_CONTACT` | Precontact reached | Shape-aware contact pose | Contact pose reached safely | 4 visits | recovery, replan, or abort |
| `PUSH_CLOSED_LOOP` | Contact or verified segment | One current-state segment | Segment or containment | 8 segments | recovery, replan, or abort |
| `VERIFY_PROGRESS` | Segment completed | Read-only feedback check | Native success or sufficient improvement | 8 checks | continue, recovery, replan, or abort |
| `RECOVER_CONTACT` | Observed loss/stagnation/rollout | Retreat and fresh plan | Revalidated plan | 2 recoveries | replan or abort |
| `REPLAN` | Recoverable rejection/drift | Fresh current-state plan | All geometry gates pass | 3 replans | abort |
| `SUCCESS` | Native stable success | None | Terminal | 1 visit | None |
| `SAFE_ABORT` | Bounded failure | None | Evidence-preserving terminal | 1 visit | None |

The executable contracts, including entry condition, target, progress
condition, timeout rule, failure reason, and allowed next states, live in
`geometry_push_types.py`. The controller rejects any undeclared transition.

## Shape-aware geometry

The desired planar direction is recomputed from the live object and region
centers:

`d = (p_target - p_object) / ||p_target - p_object||`.

For a yawed box with local half extents `(h_x, h_y)`, support along `d` is
`|d_local,x| h_x + |d_local,y| h_y`. A cylinder uses its physical planar
radius. The TCP contact center is placed behind the object by the support extent
plus a fixed gripper padding. Precontact adds a fixed free-space clearance; push
height is selected by primitive. Zero-length directions and all nonfinite or
non-positive geometry are rejected.

The final object center lies inside the native full-containment radius by a
fixed goal margin. The path is divided into bounded segments; near the region,
the maximum segment length is halved. There are no per-seed values.

## Workspace and collision feasibility

The object gate uses the complete oriented footprint against the native
workspace after subtracting object clearance and controller tracking margin.
The TCP gate subtracts its own safety clearance and the same tracking margin.
Because the native workspace is convex, checking both endpoints proves the
straight segment remains within the corresponding shrunken workspace. A plan
with a negative signed margin is rejected before execution.

Robot/link feasibility remains the responsibility of the content-bound native
Panda/MPlib collision and kinematics model at every waypoint. The geometry
layer does not claim to replace that full articulated check. A direct planner
rejection permits one bounded midpoint path; further rejection enters the
explicit replan budget rather than action repair.

## Closed-loop progress and recovery

After every short push segment the controller rereads object pose, target
distance, native success, target contact force, TCP contact distance, lateral
error, and angular speed. It never extrapolates from the source-time state.

- Native stable success terminates immediately; no further push is emitted.
- Sufficient distance reduction permits the next freshly planned segment.
- Contact loss or excessive rollout enters bounded contact recovery.
- Repeated insufficient progress enters bounded contact recovery.
- Excessive lateral drift enters bounded replanning.
- Recovery retreats through controller actions, recomputes contact from the
  live object, and reruns all geometry gates.
- Exhausted segment, state, recovery, replan, or episode budgets end in
  `SAFE_ABORT` with a classified non-success status.

The environment feedback accessor returns expert-only contact and velocity
tensors. It does not alter policy observations or task evaluation.

## Action contract

Every native action remains eight-dimensional: seven planned Panda joint
positions plus the closed-gripper command. The builder validates finiteness,
shape, and the exact native low/high bounds before execution. It rejects an
invalid action and never clips, normalizes, or repairs it. The environment
retains independent invalid-action and out-of-bounds latches.

## Evidence gates

Before any remote development evaluation, the frozen implementation must pass:

1. 10,000 randomized legal geometry fixtures, split equally across box and
   cylinder support semantics;
2. 10,000 native action constructions with zero nonfinite, shape, or bound
   violations;
3. unit tests for support geometry, arbitrary direction, zero distance,
   workspace rejection, legal transitions, bounded recovery, native success,
   seed disjointness, and sealed acceptance;
4. the 55-case historical failure regression bank;
5. the frozen 100-seed development bank at at least 95 successes with every
   zero-tolerance failure count equal to zero.

Acceptance seeds `69200..69299` cannot be loaded without a content-bound passing
development report for the exact source commit. If development fails, the
acceptance report must record zero executed episodes and the project must stop
this self-developed Push-expert line instead of starting Candidate Q.
