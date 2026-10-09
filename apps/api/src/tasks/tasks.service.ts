import { HttpException, Inject, Injectable } from '@nestjs/common';
import { startBreakthroughBattle } from '@server/combat/application/CombatV6BreakthroughService.js';
import { DRIZZLE_DATABASE } from '@server/database/database.service.js';
import type { ActiveCultivatorRef } from '@server/lib/auth/types.js';
import type { DbClient } from '@server/lib/drizzle/db.js';
import { toPlayerStateMutationResponse } from '@server/player/application/state/ResourceMutationResponse.js';
import { readResourceWithMeta } from '@server/player/application/state/ResourceReadService.js';
import { claimTaskRewardCommand } from '@server/tasks/application/TaskApplicationService.js';
import { TaskService } from '@server/tasks/application/TaskService.js';

@Injectable()
export class TasksService {
  constructor(@Inject(DRIZZLE_DATABASE) private readonly database: DbClient) {}
  async list(actor: ActiveCultivatorRef, status?: 'active' | 'completed') {
    // 破境卷宗是「懒创建」的：只有走「尝试突破」触发 getMajorBreakthroughGate
    // 时才会落库。于是境界到达圆满后打开静室，读到的任务列表仍是空的，
    // 玩家只会看到「卷宗尚未归档完成」，而且单纯刷新永远不会好。
    // 这里在读之前补一次非只读同步，把该生成的卷宗建出来。
    // 注意：readResourceWithMeta 内部是 read only 事务，写操作必须放在它外面。
    await TaskService.syncCultivatorTasks(actor.cultivatorId);
    return readResourceWithMeta(
      { kind: 'cultivator', id: actor.cultivatorId },
      'player.tasks',
      (tx) => TaskService.readCultivatorTasks(actor.cultivatorId, status, tx),
      this.database,
    );
  }

  async detail(actor: ActiveCultivatorRef, id: string) {
    const task = await TaskService.getCultivatorTask(actor.cultivatorId, id);
    if (!task) throw new HttpException({ error: '任务不存在' }, 404);
    return { success: true, data: { task } };
  }

  async challenge(actor: ActiveCultivatorRef, id: string) {
    return {
      success: true,
      data: await startBreakthroughBattle(
        { userId: actor.userId, cultivatorId: actor.cultivatorId },
        id,
      ),
    };
  }

  async claim(actor: ActiveCultivatorRef, taskId: string) {
    return toPlayerStateMutationResponse(
      await claimTaskRewardCommand({
        userId: actor.userId,
        cultivatorId: actor.cultivatorId,
        taskId,
      }),
    );
  }
}
